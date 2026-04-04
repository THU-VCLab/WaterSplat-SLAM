import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from fused_ssim import fused_ssim

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err

def simple_color_balance_tensor(image: torch.Tensor) -> torch.Tensor:
    
    r, g, b = image[0], image[1], image[2]  
    
    # 计算各通道均值
    Ravg = torch.mean(r)
    Gavg = torch.mean(g)
    Bavg = torch.mean(b)
    
    # 计算比例和饱和度级别
    Max = torch.max(torch.stack([Ravg, Gavg, Bavg]))
    ratio = Max / torch.stack([Ravg, Gavg, Bavg])
    satLevel = 0.005 * ratio
    
    # 展平各通道
    imgRGB_orig = torch.stack([r.flatten(), g.flatten(), b.flatten()])
    imRGB = torch.zeros_like(imgRGB_orig)
    
    # 对每个通道进行颜色平衡
    for ch in range(3):
        q_low = satLevel[ch].item()
        q_high = 1 - satLevel[ch].item()
        tiles = torch.quantile(imgRGB_orig[ch, :], torch.tensor([q_low, q_high], device=image.device))
        temp = torch.clamp(imgRGB_orig[ch, :], min=tiles[0], max=tiles[1])
        pmin = temp.min()
        pmax = temp.max()
        imRGB[ch, :] = (temp - pmin) * 1.0 / (pmax - pmin + 1e-8)  # 保持在 [0.0, 1.0]
    
    # 恢复原始形状 (3, H, W)
    H, W = image.shape[1], image.shape[2]
    output = torch.zeros_like(image)
    output[0] = imRGB[0].reshape(H, W)  # R
    output[1] = imRGB[1].reshape(H, W)  # G
    output[2] = imRGB[2].reshape(H, W)  # B
    
    return output.clamp(0.0, 1.0)

class DarkChannelPriorLossV3(nn.Module):
    def __init__(self, cost_ratio=1000.):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.cost_ratio = cost_ratio

    def forward(self, direct, depth=None):
        pos = self.l1(self.relu(direct), torch.zeros_like(direct))
        neg = self.smooth_l1(self.relu(-direct), torch.zeros_like(direct))
        # if (neg > 0):
        #     print(f"negative values inducing loss: {neg}")
        bs_loss = self.cost_ratio * neg + pos
        return bs_loss

class EdgeSimilarityLoss(nn.Module):
    def __init__(self):
        super(EdgeSimilarityLoss, self).__init__()

        sobel_x = torch.tensor([[[[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]]]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def rgb_to_grayscale(self, img):

        assert img.dim() == 3 and img.shape[0] == 3, f"Expected [3, H, W] tensor, got {img.shape}"
        
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        return gray.unsqueeze(0)
    def forward(self, img1, img2):

        gray1 = self.rgb_to_grayscale(img1).unsqueeze(0)  # [1, 1, H, W]
        gray2 = self.rgb_to_grayscale(img2).unsqueeze(0)  # [1, 1, H, W]

        # 计算梯度
        grad_x1 = F.conv2d(gray1, self.sobel_x, padding=1)
        grad_y1 = F.conv2d(gray1, self.sobel_y, padding=1)
        grad1 = torch.sqrt(grad_x1 ** 2 + grad_y1 ** 2 + 1e-6)

        grad_x2 = F.conv2d(gray2, self.sobel_x, padding=1)
        grad_y2 = F.conv2d(gray2, self.sobel_y, padding=1)
        grad2 = torch.sqrt(grad_x2 ** 2 + grad_y2 ** 2 + 1e-6)

        return F.l1_loss(grad1, grad2)


class GrayWorldPriorLoss(nn.Module):
    def __init__(self, target_intensity=0.5):
        super(GrayWorldPriorLoss, self).__init__()
        self.target_intensity = target_intensity

    def forward(self, J):
        assert J.dim() == 3 and J.shape[0] == 3, f"Expected [3, H, W] tensor, got {J.shape}"
        
        channel_intensities = torch.mean(J.view(3, -1), dim=1)  # 展平H,W维度后计算均值
        
        intensity_loss = (channel_intensities - self.target_intensity).pow(2).mean()
        
        if torch.isnan(intensity_loss).any() or torch.isinf(intensity_loss).any():
            print("Warning: NaN or Inf detected in intensity loss! Returning zero loss.")
            return torch.zeros_like(intensity_loss)
            
        return intensity_loss

import torch

def exposure_loss(enhanced_image, block_size=32, target_mean=0.5):
    C, H, W = enhanced_image.shape
    
    gray = 0.299 * enhanced_image[0] + 0.587 * enhanced_image[1] + 0.114 * enhanced_image[2]
    
    h_blocks = H // block_size + (1 if H % block_size != 0 else 0)
    w_blocks = W // block_size + (1 if W % block_size != 0 else 0)
    
    unfolded = gray.unfold(0, block_size, block_size).unfold(1, block_size, block_size)
    block_means = unfolded.contiguous().view(-1, block_size*block_size).mean(dim=1)
    
    # 计算损失
    loss = (block_means - target_mean).pow(2).mean()
    
    return loss

def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["training"]["alpha"] if "alpha" in config["training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_water_loss(rgb, gt_image, depth_loss=False, final=False):
    _, h, w = gt_image.shape

    ssim_lambda = 0.2

    pred_img_detach = rgb.detach()

    # RGB normalized L1
    recon_loss = torch.abs((gt_image - rgb) / (pred_img_detach + 1e-3)).mean()

    # Convert to shape [1, C, H, W] for fused SSIM
    gt_img_norm = (gt_image / (pred_img_detach + 1e-3)).unsqueeze(0)
    pred_img_norm = (rgb / (pred_img_detach + 1e-3)).unsqueeze(0)

    simloss = 1 - fused_ssim(pred_img_norm, gt_img_norm)

    main_loss = (1 - ssim_lambda) * recon_loss + ssim_lambda * simloss
    return main_loss


def get_water_loss_mask(rgb, rgb_med, water_mask, gt_image, depth_loss=False, final=False):
    _, h, w = gt_image.shape

    ssim_lambda = 0.2

    pred_img_detach = rgb.detach()

    # RGB normalized L1 with water mask separation
    recon_loss = torch.abs((gt_image - rgb_med) * water_mask / (pred_img_detach + 1e-3)).mean() + \
        torch.abs((gt_image - rgb) * ~water_mask / (pred_img_detach + 1e-3)).mean()

    # Convert to shape [1, C, H, W] for fused SSIM
    gt_img_norm = (gt_image / (pred_img_detach + 1e-3)).unsqueeze(0)
    pred_img_norm = (rgb / (pred_img_detach + 1e-3)).unsqueeze(0)

    simloss = 1 - fused_ssim(pred_img_norm, gt_img_norm)

    main_loss = (1 - ssim_lambda) * recon_loss + ssim_lambda * simloss
    return main_loss

def get_loss_mapping_all(config, outputs, viewpoint, depth_loss=False, final=False):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["training"]["rgb_boundary_threshold"]

    lambda_rgb_loss = config["training"]["rgb_boundary_threshold"]
    lambda_white_balance_loss = config["training"]["lambda_white_balance_loss"]
    lambda_bs_loss = config["training"]["lambda_bs_loss"]
    lambda_gray_loss = config["training"]["lambda_gray_loss"]

    depth_img = outputs["depth"]
    pred_img = outputs["render"] # I
    rgb_object_clr = outputs["rgb_clear"]# J
    
    gt_img_detach = gt_image.detach()
    rgb_object_clr_detach = rgb_object_clr.detach()
    # pred_img_detach = pred_img.detach()
    # depth_img_detach = depth_img.detach()

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(pred_img * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    if config["training"]["white_balance_loss"]: # weight with 0.5
        white_balance_image = simple_color_balance_tensor(rgb_object_clr_detach).detach()
        l1_white = torch.abs((white_balance_image - rgb_object_clr) / (rgb_object_clr_detach + 1e-3)).mean()  # 归一化后的L1损失
        white_balance_loss = l1_white
    else:
        white_balance_loss = torch.tensor(0.0).cuda()


    if config["training"]["bs_loss_enable"]:# and (self.step % 10 ==0) DCP and Seathru theory   self.step > 1000 weight with 0.5

        dcp_criterion = DarkChannelPriorLossV3().to("cuda")
        bsdcp_loss  = dcp_criterion(rgb_object_clr)#,depth_img_detach)
        bs_loss = bsdcp_loss
    else:
        bs_loss = torch.tensor(0.0).cuda()

    if config["training"]["gray_loss_enable"] and final: 

        edge_criterion = EdgeSimilarityLoss().to('cuda')
        edge_loss = edge_criterion(rgb_object_clr,gt_img_detach)

        gray_criterion = GrayWorldPriorLoss().to("cuda") 
        gray_loss = gray_criterion(rgb_object_clr) 

        image_enhance_loss = exposure_loss(rgb_object_clr)

        color_loss = image_enhance_loss + gray_loss + edge_loss

    else:
        color_loss = torch.tensor(0.0).cuda()


    if depth_loss:
        gt_depth = torch.from_numpy(viewpoint.depth).to(
            dtype=torch.float32, device=pred_img.device
        )[None]
        depth_pixel_mask = (gt_depth > 0.01).view(*depth_img.shape)

        l1_depth = torch.abs(depth_img * depth_pixel_mask - gt_depth * depth_pixel_mask)
        l1_depth = l1_depth.mean()
    else:
        l1_depth = torch.tensor(0.0).cuda()


    return lambda_rgb_loss*l1_rgb.mean() + lambda_white_balance_loss*white_balance_loss + lambda_bs_loss*bs_loss + lambda_gray_loss*color_loss + (1-lambda_rgb_loss)*l1_depth



def get_loss_mapping_rgb(config, image, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()

def get_loss_mapping_rgb_mask(config, image, image_med, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    image_mask = image * rgb_pixel_mask ; gt_image_mask = gt_image * rgb_pixel_mask
    image_med_mask = image_med * rgb_pixel_mask

    water_mask = viewpoint.water_mask.squeeze().unsqueeze(0)


    l1_rgb = torch.abs(image_med_mask * water_mask - gt_image_mask * water_mask) + torch.abs(image_mask * ~water_mask - gt_image_mask * ~water_mask)

    return l1_rgb.mean()



def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    if not hasattr(viewpoint, "depth") or viewpoint.depth is None:
        return torch.tensor(0.0, device=image.device, requires_grad=True)
    init_alpha = config["training"]["init_alpha"] if "init_alpha" in config["training"] else 0.95
    alpha = config["training"]["alpha"] if "alpha" in config["training"] else 0.98
    rgb_boundary_threshold = config["training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    if initialization:
        return init_alpha * l1_rgb.mean() + (1 - init_alpha) * l1_depth.mean()
    else:
        return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()
    
def get_loss_mapping_rgbd_mask(config, image, image_med, depth, viewpoint, initialization=False):
    init_alpha = config["training"]["init_alpha"] if "init_alpha" in config["training"] else 0.95
    alpha = config["training"]["alpha"] if "alpha" in config["training"] else 0.98
    rgb_boundary_threshold = config["training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]


    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    image_mask = image * rgb_pixel_mask ; gt_image_mask = gt_image * rgb_pixel_mask
    image_med_mask = image_med * rgb_pixel_mask

    water_mask = viewpoint.water_mask.squeeze().unsqueeze(0)

    
    l1_rgb = torch.abs(image_med_mask * water_mask - gt_image_mask * water_mask) + torch.abs(image_mask * ~water_mask - gt_image_mask * ~water_mask)

    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    if initialization:
        return init_alpha * l1_rgb.mean() + (1 - init_alpha) * l1_depth.mean()
    else:
        return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()



def depths_to_points(view, depthmap, world_frame):
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor([[fx, 0., W/2.], [0., fy, H/2.], [0., 0., 1.0]]).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.5, torch.arange(H, device='cuda').float() + 0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    if world_frame:
        c2w = (view.world_view_transform.T).inverse()
        rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
        rays_o = c2w[:3,3]
        points = depthmap.reshape(-1, 1) * rays_d + rays_o
    else:
        rays_d = points @ intrins.inverse().T
        points = depthmap.reshape(-1, 1) * rays_d
    return points


def depth_to_normal(view, depth, world_frame=False):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth, world_frame).reshape(*depth.shape[1:], 3)
    normal_map = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map[1:-1, 1:-1, :] = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    return normal_map, points



def get_loss_normal(depth_mean, viewpoint):
    prior_normal = viewpoint.normal.cuda()
    prior_normal = prior_normal.reshape(3, *depth_mean.shape[-2:]).permute(1,2,0)
    prior_normal_normalized = torch.nn.functional.normalize(prior_normal, dim=-1)

    normal_mean, _ = depth_to_normal(viewpoint, depth_mean, world_frame=False)
    normal_error = 1 - (prior_normal_normalized * normal_mean).sum(dim=-1)
    normal_error[prior_normal.norm(dim=-1) < 0.2] = 0
    return normal_error.mean()


def to_se3_vec(pose_mat):
    quat = R.from_matrix(pose_mat[:3, :3]).as_quat()
    return np.hstack((pose_mat[:3, 3], quat))
