# Copyright 2024 Shenyang Institute of Automation, Chinese Academy of Sciences
# Licensed under the MIT License. See the LICENSE file for details.

# The following code is licensed under the Apache License 2.0:
#
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Note: Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn

from .cudalight._torch_impl import quat_to_rotmat  # 从四元数转换为旋转矩阵的函数
from .rasterizerlight.project_gaussians import project_gaussians  # 投影高斯函数
from .rasterizerlight.rasterize import rasterize_gaussians  # 光栅化高斯函数
from .utils.sh import num_sh_bases, spherical_harmonics  # 球谐函数相关函数

from pytorch_msssim import SSIM  # 用于计算结构相似性指数
from torch.nn import Parameter  # PyTorch的参数类
from typing_extensions import Literal  # 类型提示的扩展

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.model_components.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.encodings import SHEncoding
import torch.nn.functional as F
import kornia

from test_code.loss_test import EdgeSimilarityLoss, GrayWorldPriorLoss
from test_code.loss_test import bs_dcp_loss,DeattenuateLoss,DarkChannelPriorLossV3,exposure_loss,color_balance_loss,DarkChannelPriorLossV2,GrayWorldPriorLoss,SimpleColorBalanceLoss,EdgeSimilarityLoss,UnsupervisedDarkChannelLoss # EdgeSimilarityLoss有用0.1UnsupervisedDarkChannelLoss


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


@dataclass
class LowlightUnderwaterModelConfig(ModelConfig):

    """lowlight underwater Model Config"""


    _target: Type = field(default_factory=lambda: LowlightUnderwaterModel)
    num_steps: int = 15000
    """Number of steps to train the model"""
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "black"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.5 # 影响分布,去雾选低一点，太高全没了，水下用高的可以0.5，
    """剔除高斯的不透明度阈值。可以将其设置为较低的值（例如 0.005）以获得更高的质量。"""
    cull_alpha_thresh_post: float = 0.05 # 正常光水下用高的可以0.1，
    """剔除后高斯的不透明度阈值后期的阈值"""
    reset_alpha_thresh: float = 0.5
    """重置不透明度的阈值"""
    cull_scale_thresh: float = 10.0 # 10.0 would be better？
    """剔除巨大高斯的尺度阈值"""
    continue_cull_post_densification: bool = True
    """如果为True，细化后继续裁剪高斯点"""
    zero_medium: bool = False
    """如果为True，零化介质MLP输出参数"""
    reset_alpha_every: int = 5
    """Every this many refinement steps, reset the alpha"""
    abs_grad_densification: bool = True
    """如果为True，使用绝对梯度进行密化"""
    densify_grad_thresh: float = 0.0008 #影响分布
    """用于致密高斯分布的位置梯度范数阈值 （0.0004， 0.0008）"""
    use_absgrad: bool = False # False？
    """Whether to use absgrad to densify gaussians, if False, will use grad rather than absgrad"""
    densify_size_thresh: float = 0.001
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    clip_thresh: float = 0.01
    """最小深度阈值"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 0
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 10000
    """stop splitting at this step"""
    sh_degree: int = 1
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False# False better?减少大体积的高斯分裂
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic 渲染模式将使用 EWA 卷积体积点绘制（volume splatting）与一个 [0.3, 0.3] 的屏幕空间模糊核。
    然而，这种方法不适合在高分辨率或低分辨率下渲染非常小的高斯分布（gaussians），会导致类似“别名”（aliasing）效应的伪影。
    抗锯齿模式（antialiased）通过计算补偿因子并将其应用于高斯的透明度，从而克服了这一限制，能够保持点绘制总密度的积分。
    
    然而，使用抗锯齿（antialiased）点绘制模式导出的 PLY 文件与经典模式不兼容。因此，许多为经典模式实现的 Web 查看器无法在没有修改的情况下正确渲染抗锯齿模式的 PLY 文件。
    """
    num_layers_medium: int = 2
    """Number of hidden layers for medium MLP."""
    hidden_dim_medium: int = 128
    """Dimension of hidden layers for medium MLP."""
    medium_density_bias: float = 0.0
    num_layers_color: int = 2
    """Number of hidden layers for color MLP."""
    hidden_dim_color: int = 128
    """Dimension of hidden layers for color MLP."""
    color_density_bias: float = 0.0
    """Bias for color density (sigma_bs and sigma_attn)."""
    mlp_type: Literal["tcnn", "torch"] = "tcnn"
    mlp_type2: Literal["tcnn", "torch"] = "tcnn"
    mlp_type3: Literal["tcnn", "torch"] = "tcnn"
    """Type of MLP to use for medium MLP."""
    gamma_0: float = 2.2
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SE3"))
    """Config of the camera optimizer to use"""
    use_bilateral_grid: bool = False
    """If True, use bilateral grid to handle the ISP changes in the image space. This technique was introduced in the paper 'Bilateral Guided Radiance Field Processing' (https://bilarfpro.github.io/)."""
    grid_shape: Tuple[int, int, int] = (16, 16, 8)
    """Shape of the bilateral grid (X, Y, W)"""
    color_corrected_metrics: bool = False
    """If True, apply color correction to the rendered images before computing the metrics."""
    one_color: bool = False
    """color_mlp输出为1"""
    enhance_enable: bool = False
    """启用增强,暗光必须开"""
    white_balance_loss: bool = False# 建议一直开着
    """启用白平衡loss，建议一直开着"""
    bs_loss_enable: bool = False
    """启用bs损失控制散色分量(介质)"""
    gray_loss_enable: bool = False
    """灰色世界损失"""
    water_splatting_loss: bool = False # not good enough
    """water splatting loss 关闭为原版高斯loss"""
    wb_clamp: float = 7


class LowlightUnderwaterModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: LowlightUnderwaterModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.xys_grad_norm = None
        self.sigma_activation = None
        self.colour_activation = None
        self.medium_density_bias = None
        self.color_density_bias = None
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        # 初始化方向的球面谐波编码
        # initialize the medium MLP
        self.direction_encoding = SHEncoding(levels=4, implementation="tcnn")
        # 设置颜色和sigma输出的激活函数
        self.colour_activation = nn.Sigmoid()
        self.sigma_activation = nn.Softplus()
        # medium MLP
        # 从配置中提取中间MLP的参数
        num_layers_medium=self.config.num_layers_medium,
        hidden_dim_medium=self.config.hidden_dim_medium,
        self.medium_density_bias=self.config.medium_density_bias,
        # if type is tuple, then [0]
        # 确保参数是整数或浮点数
        num_layers_medium = num_layers_medium if isinstance(num_layers_medium, int) else num_layers_medium[0]
        hidden_dim_medium = hidden_dim_medium if isinstance(hidden_dim_medium, int) else hidden_dim_medium[0]
        self.medium_density_bias = self.medium_density_bias if isinstance(self.medium_density_bias, float) else self.medium_density_bias[0]
        # color MLP
        # 从配置中提取中间MLP的参数
        num_layers_color=self.config.num_layers_color,
        hidden_dim_color=self.config.hidden_dim_color,
        self.color_density_bias=self.config.color_density_bias,
        # if type is tuple, then [0]
        # 确保参数是整数或浮点数
        num_layers_color = num_layers_color if isinstance(num_layers_color, int) else num_layers_color[0]
        hidden_dim_color = hidden_dim_color if isinstance(hidden_dim_color, int) else hidden_dim_color[0]
        self.color_density_bias = self.color_density_bias if isinstance(self.color_density_bias, float) else self.color_density_bias[0]


        # ------------------------介质网络------------------------
        # 介质 MLP
        # 如果隐藏层层数大于1
        if num_layers_medium > 1:
            self.medium_mlp = MLP(
                in_dim=self.direction_encoding.get_out_dim(), # 输入维度由 direction_encoding 模块的输出维度决定
                num_layers=num_layers_medium, # 网络的层数
                layer_width=hidden_dim_medium, # 每一层的神经元数量
                out_dim=9, # 输出维度为 9
                activation=nn.Sigmoid(), # 隐藏层的激活函数使用 Sigmoid 函数
                out_activation=None, # 输出层没有激活函数，用于回归？
                implementation=self.config.mlp_type, # 由配置参数 mlp_type 决定，可以是 "tcnn" 或 "torch"。
            )
        else:
            # 如果只指定了一层，使用线性层
            self.medium_mlp = nn.Linear(self.direction_encoding.get_out_dim(), 9)
            self.config.mlp_type = "torch"
        # ------------------------光照网络------------------------
        # 光照MLP
        # 如果隐藏层层数大于1
        if num_layers_color > 1:
            self.color_mlp = MLP(
                in_dim=self.direction_encoding.get_out_dim(),  # 输入维度由 direction_encoding 模块的输出维度决定
                num_layers=num_layers_color,  # 网络的层数
                layer_width=hidden_dim_color,  # 每一层的神经元数量
                out_dim=3,  # 输出维度为 3
                activation=nn.Sigmoid(),  # 隐藏层的激活函数使用 Sigmoid 函数
                out_activation=None,  # 输出层没有激活函数，用于回归？
                implementation=self.config.mlp_type2,  # 由配置参数 mlp_type2 决定，可以是 "tcnn" 或 "torch"。
            )
        else:
            # 如果只指定了一层，使用线性层
            self.color_mlp = nn.Linear(self.direction_encoding.get_out_dim(), 3)
            self.config.mlp_type = "torch"

        # ------------------------增强网络------------------------
        if num_layers_color > 1:
            self.enhance_mlp = MLP(
                in_dim=self.direction_encoding.get_out_dim()+self.color_mlp.out_dim,  # 输入维度由 direction_encoding 模块的输出维度决定
                num_layers=num_layers_color,  # 网络的层数
                layer_width=hidden_dim_color,  # 每一层的神经元数量
                out_dim=6,  # 输出维度为 6 alpha和gama
                activation=nn.Sigmoid(),  # 隐藏层的激活函数使用 Sigmoid 函数
                out_activation=None,  # 输出层没有激活函数，用于回归？
                implementation=self.config.mlp_type3,  # 由配置参数 mlp_type3 决定，可以是 "tcnn" 或 "torch"。
            )
        else:
            # 如果只指定了一层，使用线性层
            self.enhance_mlp = nn.Linear(self.direction_encoding.get_out_dim(), 3)
            self.config.mlp_type = "torch"

        # 根据种子点或随机值初始化高斯参数的均值
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        # 计算最近邻的距离
        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        # 计算平均距离用于缩放
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points)) #随机旋转
        dim_sh = num_sh_bases(self.config.sh_degree)

        # 如果有种子点，则初始化球面谐波特征
        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255) # 将颜色转换为SH
                shs[:, 1:, 3:] = 0.0 # 将其他维度归0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation") # 使用仅颜色优化和sigmoid激活
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :]) # 球鞋低频部分直接颜色特征，球面谐波的第 0 阶分量
            features_rest = torch.nn.Parameter(shs[:, 1:, :]) # 高阶分量（从第 1 阶到更高阶）
        else:#                  [高斯体数量，是球面谐波基数（dim_sh，根据阶数计算，RGB 通道数量]
            # 如果没有种子点，则用随机值初始化特征
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        # 初始化透明度参数
        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        # 创建高斯参数字典
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup( #closed optimizer?
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)  # 峰值信噪比
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)  # 结构相似性指数
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)  # 学习感知图像补丁相似性
        self.step = 0  # 初始化步骤计数器

        # 根据配置初始化背景颜色
        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            # 如果指定随机背景颜色，则使用默认颜色
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            # 获取指定的背景颜色
            self.background_color = get_color(self.config.background_color)
        if self.config.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                num=self.num_train_data,
                grid_X=self.config.grid_shape[0],
                grid_Y=self.config.grid_shape[1],
                grid_W=self.config.grid_shape[2],
            )

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        if self.config.sh_degree > 0:
            return self.features_dc
        else:
            return RGB2SH(torch.sigmoid(self.features_dc))

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]
    
    @property
    def medium_mlp(self):
        return self.gauss_params["medium_mlp"]

    @property
    def color_mlp(self):
        return self.gauss_params["color_mlp"]

    @property
    def enhance_mlp(self):
        return self.gauss_params["enhance_mlp"]
    
    @property
    def direction_encoding(self):
        return self.gauss_params["direction_encoding"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = self.config.num_steps
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
        Find k-nearest neighbors using sklearn's NearestNeighbors.

        Args:
            x (torch.Tensor): The data tensor of shape [num_samples, num_features]
            k (int): The number of neighbors to retrieve

        Returns:
            Tuple[np.ndarray, np.ndarray]: Distances and indices of the k-nearest neighbors
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Determine the number of samples
        n_samples = x_np.shape[0]

        # Calculate the number of neighbors, ensuring it does not exceed n_samples
        n_neighbors = min(k + 1, n_samples)

        if n_neighbors < k + 1:
            print(f"Warning: Adjusting n_neighbors from {k + 1} to {n_neighbors} because n_samples={n_samples}")

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        if n_neighbors > 1:
            # Exclude the point itself from the result and return
            return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)
        else:
            # Only one sample, no neighbors
            return distances[:, 0:0].astype(np.float32), indices[:, 0:0].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):#不是这里的问题
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        # if self.step >= self.config.stop_split_at:
        #     return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            if self.config.abs_grad_densification:
                assert self.xys_grad_abs is not None
                grads = self.xys_grad_abs.detach().norm(dim=-1)
            else:
                assert self.xys.grad is not None
                grads = self.xys.grad.detach().norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.depths_accum = self.depths
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]
                self.depths_accum[visible_mask] = self.depths[visible_mask] + self.depths_accum[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # 将所有的不透明度重置逻辑偏移 refine_every
            # 这么多步，这样我们就不会在不透明度重置时保存检查点（每2000步保存一次）
            # 然后进行修剪（cull）。只有在自上次不透明度重置之后，我们已经看到每一张图像时，才会进行分割 / 修剪。
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and (self.step % reset_interval > self.num_train_data + self.config.refine_every)
            )
            if do_densification:
                # then we densify
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])

                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()

                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                splits &= high_grads

                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads

                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # if self.step < self.config.stop_screen_size_at:
                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )                
                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None
    
            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

                # reset the exp of optimizer
                for key in ["medium_mlp", "color_mlp", "enhance_mlp","direction_encoding"]:
                    optim = optimizers.optimizers[key]
                    param = optim.param_groups[0]["params"][0]
                    param_state = optim.state[param]
                    if "exp_avg" in param_state:
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

                
            if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:                
                # Reset value is set to be reset_alpha_thresh
                reset_value = self.config.reset_alpha_thresh
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            
            self.xys_grad_norm = None
            self.vis_counts = None
            self.depths_accum = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        if self.step < self.config.stop_split_at:
            cull_alpha_thresh = self.config.cull_alpha_thresh
        else:
            cull_alpha_thresh = self.config.cull_alpha_thresh_post
        # 裁剪不透明度低于阈值的高斯点
        culls = (torch.sigmoid(self.opacities) < cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []#maybe this have some probrom
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],self.step_cb, args=[training_callback_attributes.optimizers],))
        # cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # 在每次训练迭代后执行 after_train
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, optimizers: Optimizers, step):
        self.step = step
        self.optimizers = optimizers.optimizers
    # def step_cb(self, step):
    #     self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
            "gaussian": [...],          # 高斯核参数列表（从 get_gaussian_param_groups 返回）
            "medium_mlp": [...],        # 与介质相关的 MLP 参数列表
            "color_mlp": [...],        # 与光照相关的 MLP 参数列表
            "enhance_mlp": [...],        # 与增强相关的 MLP 参数列表
            "direction_encoding": [...] # 与方向编码相关的参数列表
        """
        gps = self.get_gaussian_param_groups()
        gps["medium_mlp"] = list(self.medium_mlp.parameters())
        # print(gps["medium_mlp"])
        gps["color_mlp"] = list(self.color_mlp.parameters())
        gps["enhance_mlp"] = list(self.enhance_mlp.parameters())
        gps["direction_encoding"] = list(self.direction_encoding.parameters())
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    def get_outputs(self, camera: Cameras,obb_box: Optional[OrientedBox] = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.
            obb_box (Optional[OrientedBox], optional): 可选的定向盒子，用于裁剪。默认为None。

        Returns:
            Dict[str, Union[torch.Tensor, List]]: 渲染结果字典。
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        camera_downscale = self._get_downscale_factor()# 下采样4倍
        camera.rescale_output_resolution(1 / camera_downscale)
        # 获取相机的旋转和平移矩阵
        R = camera.camera_to_worlds[0, :3, :3]  # 3x3旋转矩阵
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3x1平移向量

        # 翻转z和y轴以与gsplat的约定对齐
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit  # 应用翻转
        # 计算世界到相机的矩阵
        R_inv = R.T  # 旋转矩阵的逆等于其转置
        T_inv = -R_inv @ T  # 计算平移的逆
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)  # 创建4x4单位矩阵
        viewmat[:3, :3] = R_inv  # 设置旋转部分
        viewmat[:3, 3:4] = T_inv  # 设置平移部分
        # 计算相机的视场角
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)  # 记录图像尺寸
        self.last_fx = camera.fx.item()  # 记录焦距fx
        self.last_fy = camera.fy.item()  # 记录焦距fy

        # 介质部分
        # 像素编码
        y = torch.linspace(0., H, H, device=self.device)
        x = torch.linspace(0., W, W, device=self.device)
        yy, xx = torch.meshgrid(y, x)
        yy = (yy - cy) / camera.fy.item()
        xx = (xx - cx) / camera.fx.item()
        directions = torch.stack([yy, xx, -1 * torch.ones_like(xx)], dim=-1)
        norms = torch.linalg.norm(directions, dim=-1, keepdim=True)
        directions = directions / norms
        directions = directions @ R

        directions_flat = directions.view(-1, 3)
        directions_encoded = self.direction_encoding(directions_flat)# 球谐编码，编码之后是H*W,16
        outputs_shape = directions.shape[:-1]

        # 介质MLP的前向传递
        if self.config.mlp_type == "tcnn":
            medium_base_out = self.medium_mlp(directions_encoded)
        else:
            medium_base_out = self.medium_mlp(directions_encoded.float())

        # 不同输出的激活函数
        medium_rgb = (#[H,W,3] CLAMP(0-1)
            self.colour_activation(medium_base_out[..., :3])
            .view(*outputs_shape, -1)
            .to(directions)
        )# 颜色部分应用Sigmoid激活
        medium_bs = (#[H,W,3] CLAMP(>0)
            self.sigma_activation(medium_base_out[..., 3:6] + self.medium_density_bias)
            .view(*outputs_shape, -1)
            .to(directions)
        )# sigma_bs部分应用Sigmoid激活
        medium_attn = (#[H,W,3] CLAMP(>0)
            self.sigma_activation(medium_base_out[..., 6:] + self.medium_density_bias)
            .view(*outputs_shape, -1)
            .to(directions)
        )# sigma_attn部分应用Sigmoid激活

        if self.config.zero_medium:
            medium_rgb = torch.zeros_like(medium_rgb)
            medium_bs = torch.zeros_like(medium_bs)
            medium_attn = torch.zeros_like(medium_attn)

        # 颜色MLP的前向传递
        if self.config.mlp_type2 == "tcnn":
            color_base_out = self.color_mlp(directions_encoded)
        else:
            color_base_out = self.color_mlp(directions_encoded.float())

        # 不同输出的激活函数
        color_rgb = (  # [H,W,3] CLAMP(0-1)
            self.colour_activation(color_base_out[..., :3])
            .view(*outputs_shape, -1)
            .to(directions)
        )  # 颜色部分应用Sigmoid激活
        color_rgb_reshaped = color_rgb.view(-1, color_rgb.shape[-1])
        # 增强MLP的前向传递
        if self.config.mlp_type3 == "tcnn":
            enhance_base_out = self.enhance_mlp(torch.cat([directions_encoded, color_rgb_reshaped], dim=-1))
        else:
            enhance_base_out = self.enhance_mlp(torch.cat([directions_encoded, color_rgb_reshaped], dim=-1).float())

        # 不同输出的激活函数
        gamma = (  # [H,W,3] CLAMP(0-1)
            self.colour_activation(enhance_base_out[..., :3] )
            .view(*outputs_shape, -1)
            .to(directions)
        )  # gamma部分应用Sigmoid激活
        gamma = torch.clamp(gamma, min=1e-4, max=1.0)  # 确保gamma的范围在[1e-4, 1.0]
        alpha = (  # [H,W,3] CLAMP(>0)
            self.colour_activation(enhance_base_out[..., 3:6] )
            .view(*outputs_shape, -1)
            .to(directions)
        )  # alpha部分应用Sigmoid激活
        gamma = torch.clamp(gamma, min=1e-4, max=1.0)  # 避免gamma过小或过大
        alpha = torch.clamp(alpha, min=1e-4)  # 避免alpha过小
        color_rgb = torch.clamp(color_rgb, min=1e-4, max=1.0)  # 避免color_rgb为0
        final_gamma = torch.clamp(1 / (gamma + self.config.gamma_0), min=0.1, max=5)  # 限制final_gamma范围
        assert not torch.isnan(gamma).any(), "NaN detected in gamma!"
        assert not torch.isnan(alpha).any(), "NaN detected in alpha!"
        assert not torch.isnan(final_gamma).any(), "NaN detected in final_gamma!"

        enhanced_color_rgb = (color_rgb / (alpha + 1e-4)) ** final_gamma
        enhanced_color_rgb = torch.clamp(enhanced_color_rgb, min=-10, max=10)  # 限制输入Sigmoid的范围

        # enhanced_color_rgb = torch.sigmoid(enhanced_color_rgb)*1.5

        # print("enhanced_color_rgb min/max:", enhanced_color_rgb.min().item(), enhanced_color_rgb.max().item())
        # if self.step<5000:
        #     enhanced_color_rgb = color_rgb
        # else:
        #     enhanced_color_rgb = color_rgb*1.5

        if self.config.one_color: # without enhance out put
            color_rgb = torch.ones_like(color_rgb)
            enhanced_color_rgb = torch.ones_like(color_rgb)

        # cropping
        if self.crop_box is not None and not self.training: # 如果有裁剪盒子且不在训练模式下
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0: # 裁剪盒子内没有点，就返回返回背景色
                rgb = medium_rgb
                depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": medium_rgb,
                        "rgb_object": torch.zeros_like(rgb), "rgb_medium": medium_rgb, "pred_image": rgb,
                        "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn, "enhanced_color_rgb": enhanced_color_rgb}
        else:
            crop_ids = None # 没有裁剪

        if crop_ids is not None and crop_ids.sum() != 0:# 裁剪盒子内有点,有高斯点在裁剪盒子内，选择这些高斯点
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:# 否则，选择所有高斯点
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)# 组合球鞋,需要始终改为0阶段分量，用mlp的结果弥补
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        # 进行3D到2D投影，获得屏幕上的高斯体信息，位置，深度...
        self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
            clip_thresh=self.config.clip_thresh,
        )  # type: ignore

        self.depths = depths.detach()

        # 在返回之前将相机重新缩放回原始尺寸
        camera.rescale_output_resolution(camera_downscale)
        # 如果没有高斯点被渲染，返回背景色
        if (self.radii).sum() == 0:
            rgb = medium_rgb
            depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
            return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": medium_rgb,
                    "rgb_object": torch.zeros_like(rgb), "rgb_clear": torch.zeros_like(rgb),  "rgb_medium": medium_rgb, "pred_image": rgb,
                    "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn,"color_rgb":color_rgb, "enhanced_color_rgb": enhanced_color_rgb}

        if self.config.sh_degree > 0: # 如果使用球谐函数，计算视线方向并编码
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)  # 归一化视线方向
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)  # 计算当前使用的球谐阶数
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)  # 计算球谐函数颜色
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # 调整颜色范围
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        opacities = None
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        self.xys_grad_abs = torch.zeros_like(self.xys)
        # rasterize_gaussians重要的光删函数
        rgb_object, rgb_clear, rgb_medium, depth_im, alpha = rasterize_gaussians(  # type: ignore
            self.xys, # 前面光栅化传过来的参数
            self.xys_grad_abs, # 每一次训练初始化为0
            depths, # 前面光栅化传过来的深度
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            medium_rgb,
            medium_bs,
            medium_attn,
            # color_rgb,
            enhanced_color_rgb,
            H,
            W,
            BLOCK_WIDTH,
            background=medium_rgb,
            return_alpha=True,
            step=self.step,
        )  # type: ignore
        rgb_object = torch.clamp(rgb_object,0,1)
        rgb = rgb_object + rgb_medium # rgb_object[0-1],rgb_medium[0-1]
        rgb = torch.clamp(rgb,0,1)
        rgb_clear = torch.clamp(rgb_clear, 0., 1.)#[R,G,B]
        depth_im = depth_im[..., None]
        alpha = alpha[..., None]
        depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())

        return {"rgb": rgb, "depth": depth_im, "accumulation": alpha, "background": medium_rgb,
                "rgb_object": rgb_object, "rgb_clear": rgb_clear,  "rgb_medium": rgb_medium, "pred_image": rgb,
                "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn,"color_rgb":color_rgb, "enhanced_color_rgb": enhanced_color_rgb}  # type: ignore

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        if self.config.enhance_enable: #:  # 启用增强，增加白平衡处理gt，
            gt_img = self.shades_of_grey_white_balance_torch(gt_img, p=1, gamma=1.2, clamp_max=self.config.wb_clamp)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            # alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            # return alpha * image[..., :3] + (1 - alpha) * background
            return image[..., :3]# important 取消了原版高斯的背景混合方式
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]

        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.config.color_corrected_metrics:  # 颜色矫正指标为ture
            cc_rgb = color_correct(predicted_rgb, gt_rgb) # 使预测的图接近gt图的颜色，然后检测psnr指标
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb) #

        metrics_dict["gaussian_count"] = self.num_points
        for i in range(3):
            # 3 channels
            metrics_dict[f"medium_attn_{i}"] = outputs["medium_attn"][:, :, i].mean()
            metrics_dict[f"medium_bs_{i}"] = outputs["medium_bs"][:, :, i].mean()
            metrics_dict[f"medium_rgb_{i}"] = outputs["medium_rgb"][:, :, i].mean()
            metrics_dict[f"color_rgb_{i}"] = outputs["color_rgb"][:, :, i].mean()
            metrics_dict[f"enhanced_color_rgb{i}"] = outputs["enhanced_color_rgb"][:, :, i].mean()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        depth_img = outputs["depth"]
        pred_img = outputs["rgb"] # I
        rgb_object = outputs["rgb_object"]# D
        rgb_object_clr = outputs["rgb_clear"]# J
        medium_rgb = outputs["medium_rgb"]# B
        gt_img_detach = gt_img.detach()
        rgb_object_clr_detach = rgb_object_clr.detach()
        pred_img_detach = pred_img.detach()
        depth_img_detach = depth_img.detach()

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask
        # 损失用来减少大体积的高斯分裂
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        if self.config.white_balance_loss: # weight with 0.5
            white_balance_image = self.simple_color_balance_tensor(rgb_object_clr_detach).detach()
            l1_white = torch.abs((white_balance_image - rgb_object_clr) / (rgb_object_clr_detach + 1e-3)).mean()  # 归一化后的L1损失
            white_balance_loss = l1_white * 0.1
        else:
            white_balance_loss = torch.tensor(0.0).to(self.device)
        # bs（DCP）理论用于约束rgb_clear
        if self.config.bs_loss_enable:# and (self.step % 10 ==0) DCP and Seathru theory   self.step > 1000 weight with 0.5

            dcp_criterion = DarkChannelPriorLossV3().to("cuda")
            bsdcp_loss  = dcp_criterion(rgb_object_clr) * 0.08#,depth_img_detach)# bsdcp_loss必须有gt约束才好用
            bs_loss = bsdcp_loss
        else:
            bs_loss = torch.tensor(0.0).to(self.device)
        # 增强用的灰度世界和梯度保边loss（控制enhance的图像也就是无衰减无介质的clr图）开启增强才启用
        if self.config.gray_loss_enable and self.step >3000: # weight with 1

            edge_criterion = EdgeSimilarityLoss().to('cuda') # 有约束好使
            edge_loss = edge_criterion(rgb_object_clr,gt_img_detach)*0.1

            gray_criterion = GrayWorldPriorLoss().to("cuda") # 颜色对了！！！！work
            gray_loss = gray_criterion(rgb_object_clr)*0.1 #可选 gray_loss = color_balance_loss(rgb_object_clr)

            image_enhance_loss = exposure_loss(rgb_object_clr)*0.1

            color_loss = image_enhance_loss + gray_loss + edge_loss

        else:
            color_loss = torch.tensor(0.0).to(self.device)
        # 原版water splatting loss，不启用使用gsplat的loss
        if self.config.water_splatting_loss:# and self.step<5000:
            recon_loss = torch.abs((gt_img - pred_img) / (pred_img_detach + 1e-3)).mean()  # 归一化后的L1损失
            simloss = 1 - self.ssim((gt_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...], (pred_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...])
            main_loss = (1 - self.config.ssim_lambda) * recon_loss + self.config.ssim_lambda * simloss
        else:
            Ll1 = torch.abs(gt_img - pred_img).mean()
            simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])  # add bs loss and other loss bs_loss = BackscatterLoss(rgb_object)  da_loss = DeattenuateLoss(rgb_object, rgb_clear)# 有问题!!!!!!
            main_loss = (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss
        loss_dict = {
            "main_loss": main_loss,
            "bs_loss":bs_loss,
            "scale_reg": scale_reg,
            "color_loss": color_loss,
            "white_balance_loss": white_balance_loss
        }

        if self.training: # 应该也不是这的问题
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        if not self.training:
            with torch.no_grad():
                outs = self.get_outputs(camera.to(self.device), obb_box=obb_box)
        else:
            outs = self.get_outputs(camera.to(self.device), obb_box=obb_box)
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        # predicted_rgb = outputs["rgb_clear"]
        cc_rgb = None

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        output_gt_rgb = gt_rgb.cpu()

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        if self.config.color_corrected_metrics:
            assert cc_rgb is not None
            cc_psnr = self.psnr(gt_rgb, cc_rgb)
            cc_ssim = self.ssim(gt_rgb, cc_rgb)
            cc_lpips = self.lpips(gt_rgb, cc_rgb)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
            metrics_dict["cc_ssim"] = float(cc_ssim)
            metrics_dict["cc_lpips"] = float(cc_lpips)

        images_dict = {"gt": output_gt_rgb, "rgb_medium": outputs["rgb_medium"], "rgb_object": outputs["rgb_object"],
                       "depth": outputs["depth"], "rgb": outputs["rgb"], "rgb_clear": outputs["rgb_clear"]}

        return metrics_dict, images_dict

    def gamma_correction_torch(self, image: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        # 应用伽马校正
        inv_gamma = 1.0 / gamma
        image = torch.clamp(image, min=0.0, max=1.0)
        corrected_image = torch.pow(image, inv_gamma)
        # 如果原始图像的最大值大于 1，则将结果缩放回 [0, 255]
        if image.max() > 1.0:
            corrected_image = corrected_image * 255.0
        return corrected_image

    def compute_minkowski_norm_torch(self, image: torch.Tensor, p: float) -> torch.Tensor:
        if p == float('inf'):
            return torch.amax(image, dim=(0, 1))
        else:
            return torch.mean(torch.pow(image, p), dim=(0, 1)) ** (1.0 / p)

    def adjust_lambda_torch(self, normal_color_count: float, lambda_default: float = 0.2, lambda_max: float = 0.5) -> float:
        if normal_color_count == 0:
            return lambda_max
        # 线性反比调整 λ
        lambda_val = -normal_color_count + lambda_max + lambda_default
        # 限制 λ 在 [0, lambda_max] 范围内
        lambda_val = max(0.0, min(lambda_val, lambda_max))
        return lambda_val

    def shades_of_grey_white_balance_torch(self, image: torch.Tensor, p: float = 1, lambda_default: float = 0.2,
                                           gamma: float = 1.2, clamp_max: float = 7) -> torch.Tensor:
        assert not image.requires_grad, "Gradient tracking is enabled for 'image'"

        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        if gamma != 1.0:
            image = self.gamma_correction_torch(image, gamma=gamma)
        mu_ref = self.compute_minkowski_norm_torch(image, p)  # 形状为 [3]
        hist_size = 32
        hist_range = [0.0, 1.0]
        bins = torch.linspace(hist_range[0], hist_range[1], hist_size + 1, device=image.device)
        # 计算每个通道的直方图
        histograms = []
        for c in range(3):  # 对于每个通道
            channel = image[:, :, c]
            hist = torch.histc(channel, bins=hist_size, min=hist_range[0], max=hist_range[1])
            histograms.append(hist)
        # 计算颜色数量
        color_count = sum([torch.sum(hist > 0).item() for hist in histograms])
        # 归一化颜色数量
        normal_color_count = color_count / (hist_size * 3)
        # 调整 λ
        lambda_val = self.adjust_lambda_torch(normal_color_count, lambda_default=lambda_default)
        # 估算光照 μI
        mu_I = 0.5 + lambda_val * mu_ref  # μI 应在 [0.5, 1] 范围内
        scale = (mu_I / (mu_ref + 1e-6)) * 0.7  # 避免除以零
        scale = torch.clamp(scale, min=0.1, max=clamp_max)  # 限制缩放因子


        balanced_image = image * scale.view(1, 1, 3)
        balanced_image = torch.clamp(balanced_image, 0.0, 1.0)
        if image.max() > 1.0:
            balanced_image = balanced_image * 255.0

        # import matplotlib.pyplot as plt
        # _image = image.detach().cpu().numpy()
        # plt.subplot(1, 2, 1)
        # plt.imshow(_image)
        # plt.title("Original Image")
        # plt.axis("off")
        #
        # _balanced_image = balanced_image.detach().cpu().numpy()
        # plt.subplot(1,2,2)
        # plt.imshow(_balanced_image)
        # plt.title("balance Image without clamp")
        # plt.axis("off")
        # plt.show()
        return balanced_image

    @medium_mlp.setter
    def medium_mlp(self, value):
        self._medium_mlp = value

    @color_mlp.setter
    def color_mlp(self, value):
        self._color_mlp = value

    def simple_color_balance_tensor(self, image: torch.Tensor) -> torch.Tensor:
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        Bavg = torch.mean(b)
        Gavg = torch.mean(g)
        Ravg = torch.mean(r)
        Max = torch.max(torch.stack([Bavg, Gavg, Ravg]))
        ratio = Max / torch.stack([Bavg, Gavg, Ravg])
        satLevel = 0.005 * ratio
        imgRGB_orig = torch.stack([b.flatten(), g.flatten(), r.flatten()])
        imRGB = torch.zeros_like(imgRGB_orig)
        for ch in range(3):
            q_low = satLevel[ch].item()
            q_high = 1 - satLevel[ch].item()
            tiles = torch.quantile(imgRGB_orig[ch, :], torch.tensor([q_low, q_high], device=image.device))
            temp = torch.clamp(imgRGB_orig[ch, :], min=tiles[0], max=tiles[1])
            pmin = temp.min()
            pmax = temp.max()
            imRGB[ch, :] = (temp - pmin) * 1.0 / (pmax - pmin + 1e-8)  # 保持在 [0.0, 1.0]
        H, W, _ = image.shape
        output = torch.zeros_like(image)
        output[:, :, 0] = imRGB[0, :].reshape(H, W)
        output[:, :, 1] = imRGB[1, :].reshape(H, W)
        output[:, :, 2] = imRGB[2, :].reshape(H, W)
        output = output.clamp(0.0, 1.0)
        return output

    """""------------------------------"""

    def min_dark_channel(self, img: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
        dark = img.min(dim=1, keepdim=True)[0]  # [1, 1, H, W]
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=img.device)
        dark_eroded = F.conv2d(dark, kernel, padding=kernel_size // 2, groups=1)
        dark_eroded = dark_eroded / (kernel_size * kernel_size)
        dark_dilated = -F.max_pool2d(-dark_eroded, kernel_size, stride=1, padding=kernel_size // 2)
        return dark_dilated

    def mid_dark_channel(self, img: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
        dark = img.min(dim=1, keepdim=True)[0]  # [1, 1, H, W]
        dark_median = kornia.filters.median_blur(dark, kernel_size=kernel_size)
        return dark_median

    def kmeans_dark_channel(self, img: torch.Tensor, kernel_size: int = 15, k: int = 8, max_iters: int = 10) -> torch.Tensor:
        B, C, H, W = img.shape
        Z = img.view(B, C, -1).permute(0, 2, 1)  # [1, H*W, 3]

        # 初始化聚类中心（随机选择 k 个像素）
        indices = torch.randperm(H * W)[:k]
        centers = Z[0, indices, :].clone()  # [k, 3]

        for _ in range(max_iters):
            # 计算每个像素到每个中心的距离
            distances = torch.cdist(Z, centers.unsqueeze(0), p=2)  # [1, H*W, k]

            # 分配每个像素到最近的中心
            labels = distances.argmin(dim=2)  # [1, H*W]

            # 重新计算聚类中心
            new_centers = []
            for i in range(k):
                mask = (labels == i).float().unsqueeze(2)  # [1, H*W, 1]
                if mask.sum() == 0:
                    new_centers.append(centers[i])
                else:
                    new_center = (Z * mask).sum(dim=1) / mask.sum()
                    new_centers.append(new_center[0])
            centers = torch.stack(new_centers)  # [k, 3]

        # 分配每个像素到对应的聚类中心
        res = centers[labels.squeeze(0)]  # [H*W, 3]
        res2 = res.view(B, C, H, W)  # [1, 3, H, W]

        # 计算每个像素的最小值
        dark = res2.min(dim=1, keepdim=True)[0]  # [1, 1, H, W]

        # 定义结构元素（矩形）
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=img.device)

        # 腐蚀操作
        dark_eroded = F.conv2d(dark, kernel, padding=kernel_size // 2, groups=1)
        dark_eroded = dark_eroded / (kernel_size * kernel_size)

        return dark_eroded

    def add_dark_channels(self, dark1: torch.Tensor, dark2: torch.Tensor, beta: float = 0.7) -> torch.Tensor:
        return beta * dark1 + (1 - beta) * dark2

    def gauss_add(self, Fdark: torch.Tensor, mindark: torch.Tensor, a1: float = 4, a2: float = 2,
                  z: float = 0.6) -> torch.Tensor:
        gx = a2 * torch.exp(-((1.5 - Fdark) ** 2) / z)
        Tx = gx + a1
        Ig = (a1 * Fdark + gx * mindark) / Tx
        return Ig

    def light_channel(self, img: torch.Tensor, dark: torch.Tensor, rate: float = 0.001) -> torch.Tensor:
        B, C, H, W = img.shape
        num_pixels = max(int(H * W * rate), 1)  # 选取前0.001的像素

        # 展平暗通道
        flat_dark = dark.view(B, -1)  # [1, H*W]

        # 获取最暗的 num_pixels 个索引
        _, indices = torch.topk(flat_dark, num_pixels, dim=1, largest=True, sorted=False)  # [1, num_pixels]

        # 获取这些像素的亮度
        brightest = img.view(B, C, -1).permute(0, 2, 1).gather(1, indices.unsqueeze(-1).repeat(1, 1,
                                                                                               C))  # [1, num_pixels, 3]

        # 计算每个像素的亮度（均值）
        brightness = brightest.mean(dim=2)  # [1, num_pixels]

        # 选择最亮的像素作为大气光
        max_brightness, max_idx = brightness.max(dim=1)  # [1], [1]
        A = brightest[0, max_idx, :].unsqueeze(0)  # [1, 3]

        return A

    def transmission_estimate(self, img: torch.Tensor, A: torch.Tensor, dark_channel: torch.Tensor, omega: float = 0.95,
                              size: int = 15) -> torch.Tensor:
        # 确保 A 的形状为 [1, 3, 1, 1]
        A = A.view(1, 3, 1, 1)  # [1, 3, 1, 1]

        # 归一化图像
        norm_img = img / A  # [1, 3, H, W]

        # 计算暗通道
        min_dc = self.min_dark_channel(norm_img, size)  # [1, 1, H, W]

        # 估计透射率
        transmission = 1 - omega * min_dc
        transmission = torch.clamp(transmission, 0, 1)

        return transmission

    def guided_filter(self, img: torch.Tensor, p: torch.Tensor, r: int = 60, eps: float = 1e-4) -> torch.Tensor:
        mean_guide = kornia.filters.box_blur(img, kernel_size=(r, r))
        mean_p = kornia.filters.box_blur(p, kernel_size=(r, r))
        mean_guide_p = kornia.filters.box_blur(img * p, kernel_size=(r, r))
        cov_guide_p = mean_guide_p - mean_guide * mean_p

        mean_guide_sq = kornia.filters.box_blur(img * img, kernel_size=(r, r))
        var_guide = mean_guide_sq - mean_guide * mean_guide

        a = cov_guide_p / (var_guide + eps)
        b = mean_p - a * mean_guide

        mean_a = kornia.filters.box_blur(a, kernel_size=(r, r))
        mean_b = kornia.filters.box_blur(b, kernel_size=(r, r))

        q = mean_a * img + mean_b
        return q

    def transmission_refine(self, img: torch.Tensor, transmission: torch.Tensor, r: int = 60,
                            eps: float = 1e-4) -> torch.Tensor:
        gray = kornia.color.rgb_to_grayscale(img)  # [1, 1, H, W]

        refined_transmission = self.guided_filter(gray, transmission, r=r, eps=eps)
        refined_transmission = torch.clamp(refined_transmission, 0, 1)

        return refined_transmission

    def recover_image(self, img: torch.Tensor, transmission: torch.Tensor, A: torch.Tensor, tx: float = 0.1) -> torch.Tensor:
        t = torch.clamp(transmission, min=tx)  # [1, 1, H, W]
        recovered = (img - A.view(1, 3, 1, 1)) / t + A.view(1, 3, 1, 1)
        recovered = torch.clamp(recovered, 0, 1)
        return recovered

    def bs_dcp(self, image, beta: float = 0.7, k: int = 8, rate: float = 0.001, size: int = 15):

        img = image  # [H, W, 3]
        if img.ndim == 2:
            img = img.unsqueeze(-1).repeat(1, 1, 3)  # 转换为 RGB
        img = img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        min_dc = self.min_dark_channel(img, size)  # [1, 1, H, W]
        mid_dc = self.mid_dark_channel(img, size)  # [1, 1, H, W]
        kmeans_dc = self.kmeans_dark_channel(img, size, k)  # [1, 1, H, W]

        added_dark = self.add_dark_channels(mid_dc, kmeans_dc, beta)  # [1, 1, H, W]
        combined_dark = self.gauss_add(added_dark, min_dc)  # [1, 1, H, W]

        A = self.light_channel(img, combined_dark, rate)  # [1, 3]

        transmission = self.transmission_estimate(img, A, combined_dark, omega=0.95, size=size)  # [1, 1, H, W]

        refined_transmission = self.transmission_refine(img, transmission)  # [1, 1, H, W]

        result = self.recover_image(img, refined_transmission, A, tx=0.1)  # [1, 3, H, W]
        result = result.squeeze(0).permute(1, 2, 0)
        # import matplotlib.pyplot as plt
        # _image = image.detach().cpu().numpy()
        # plt.subplot(1, 2, 1)
        # plt.imshow(_image)
        # plt.title("Original Image")
        # plt.axis("off")
        #
        # dcp_image = result.detach().cpu().numpy()
        # plt.subplot(1,2,2)
        # plt.imshow(dcp_image)
        # plt.title("dcp image")
        # plt.axis("off")
        # plt.show()

        return result

    """----------------------------------"""

    def grad(self,img):# wrong
        img = img.permute(2, 0, 1)  # 调换维度顺序 -> [3, H, W]
        # img = img.unsqueeze(0)  # 增加 batch 维度 -> [1, 3, H, W]
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)[None, None, :].expand(1, 3, -1, -1).cuda()

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32)[None, None, :].expand(1, 3, -1, -1).cuda()
        grad_x = F.conv2d(img[None, :], sobel_x)
        grad_y = F.conv2d(img[None, :], sobel_y)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return grad

    def get_grad_loss(self,img1, img2):
        """
        计算两个图像的梯度损失。

        Args:
            grad_fn (Gradient): 梯度计算模块。
            img1 (torch.Tensor): 第一个图像张量，形状为 [B, 3, H, W]。
            img2 (torch.Tensor): 第二个图像张量，形状为 [B, 3, H, W]。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 标准化后的梯度张量。
        """
        g1 = self.grad(img1)
        g2 = self.grad(img2)
        scale = img1.mean() / img2.mean()
        g1 = g1
        g2 = g2 * scale
        g1 = g1.repeat(1, 3, 1, 1)  # [1, 3, H, W]
        g2 = g2.repeat(1, 3, 1, 1)  # [1, 3, H, W]
        return 0.1* (1.0 - self.ssim(g1, g2))
    """----------------------------------"""

    @direction_encoding.setter
    def direction_encoding(self, value):
        self._direction_encoding = value

    @enhance_mlp.setter
    def enhance_mlp(self, value):
        self._enhance_mlp = value


def get_gray_loss(enhance_pooled):
    """
    计算图像的灰度损失（示例实现）。

    Args:
        enhance_pooled (torch.Tensor): 处理后的图像张量，形状为 [B, 3, H, W]。

    Returns:
        torch.Tensor: 灰度损失标量。
    """
    # 计算每个像素的通道标准差，鼓励图像趋向于灰度图
    gray_std = enhance_pooled.std(dim=1, keepdim=True)  # [B, 1, H, W]
    return gray_std.mean()

def compute_color_loss(enhance, get_gray_loss, fixed_exposure=0.8, exposure_loss_lambda=0.1, gray_loss_lambda=0.5):
    """
    计算颜色损失，包括梯度损失、曝光损失和灰度损失。

    Args:
        outputs (dict): 模型输出，包含键 "rgb_clear_clamp"。
        gt_img (torch.Tensor): 目标图像张量，形状为 [H, W, 3]。
        ssim (callable): SSIM 损失函数。
        get_gray_loss (callable): 灰度损失函数。
        grad_fn (Gradient): 梯度计算模块。
        fixed_exposure (float): 固定曝光值。
        grad_loss_lambda (float): 梯度损失权重。
        exposure_loss_lambda (float): 曝光损失权重。
        gray_loss_lambda (float): 灰度损失权重。

    Returns:
        torch.Tensor: 总颜色损失标量。
    """

    if enhance.ndim == 3:
        enhance = enhance.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # 计算曝光损失
    # 平均池化以减少噪声
    enhance_pooled = nn.functional.avg_pool2d(enhance, kernel_size=5)  # [1, 3, H//5, W//5]
    # 计算每个通道的均值
    exposure = enhance_pooled.mean(dim=(2, 3))  # [1, 3]
    # 计算曝光偏差的均方误差
    exposure_loss = ((exposure - fixed_exposure) ** 2).mean()  # [1]
    image_enhance_loss = exposure_loss_lambda * exposure_loss  # [1]

    # 计算灰度损失
    gray_loss = gray_loss_lambda * get_gray_loss(enhance_pooled)  # [1]

    # 灰度和增强损失
    color_loss = gray_loss + image_enhance_loss
    return color_loss
