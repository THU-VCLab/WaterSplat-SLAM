#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import numpy as np
import open3d as o3d
from typing import Optional
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gaussian.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian.utils.sh_utils import RGB2SH
from mast3r_slam.network import MLP, Medium, Light, BRDF
import torch.optim as optim


class GaussianModel:
    def __init__(self, sh_degree: int, embedded_dim = 128, L_degree = 4, layer_num = 4, config=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self.max_radii2D = torch.empty(0, device="cuda")
        self.xyz_gradient_accum = torch.empty(0, device="cuda")
        self.depths_accum = torch.empty(0, 1, device="cuda")


        self.unique_kfIDs = torch.empty(0).int()
        self.n_obs = torch.empty(0).int()

        self.L_degree = L_degree

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize




        self.Medium_MLP = Medium(levels=4, hidden_dim=128, num_layers=4, mlp_type="tcnn", density_bias=0.0) # or torch
        self.Light_MLP = Light(levels=4, hidden_dim=128, num_layers=4, mlp_type="tcnn")

        # self.use_brdf = config.get("Training", {}).get("use_brdf", False)
        # if self.use_brdf:
        # self.BRDF_model = BRDF(hidden_dim=embedded_dim, num_layers=layer_num, implementation="tcnn")

        # self.Medium_optimizer = optim.Adam(
        #     list(self.Medium_MLP.parameters()) + 
        #     list(self.Light_MLP.parameters()) +
        #     list(self.BRDF_model.parameters()),
        #     lr=config["opt_params"]["medium_lr"]
        # )
       

        # self.Medium_optimizer = optim.Adam(
        #     list(self.Medium_MLP.parameters()) +
        #     list(self.Light_MLP.parameters()) ,
        #     lr=config["opt_params"]["medium_lr"]
        # )
        self.Medium_optimizer = optim.Adam(
            list(self.Medium_MLP.parameters()) ,
            lr=config["opt_params"]["medium_lr"],
            eps = config["opt_params"]["medium_eps"],
        )

        self.config = config
        self.ply_input = None

        self.isotropic = False

    def compute_gaussian_MB(self):
        #计算当前gs的显存占用（MB)
         
        total_bytes = 0
        # 需要检查的张量属性列表
        tensor_attrs = [
            '_xyz', '_features_dc', '_features_rest', 
            '_scaling', '_rotation', '_opacity',
            'max_radii2D', 'xyz_gradient_accum', 
            'depths_accum', 'unique_kfIDs', 'n_obs'
        ]
        
        for attr_name in tensor_attrs:
            tensor = getattr(self, attr_name, None)
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                total_bytes += tensor.numel() * tensor.element_size()
        
        # 转换为MB (1 MB = 1048576 bytes)
        gaussian_size = self._xyz.shape[0]
        return gaussian_size, total_bytes / 1048576
        

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc # (N, 1, 3)
        features_rest = self._features_rest # （N, (最大球谐阶数 + 1)² - 1, 3）
        return torch.cat((features_dc, features_rest), dim=1) # (N, (最大球谐阶数 + 1)², 3)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None, gsplat=True):
        cam = cam_info
        rgb_raw = (cam.original_image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
        depth = o3d.geometry.Image(depthmap.astype(np.float32))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init, gsplat)

    def create_pcd_from_image_render(self, cam_info, ext_points, init=False, scale=2.0, depthmap=None, gsplat=True):
        cam = cam_info
        rgb_raw = (cam.original_image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
        depth = o3d.geometry.Image(depthmap.astype(np.float32))

        return self.create_pcd_from_image_and_depth_render(cam, ext_points, rgb, depth, init, gsplat=gsplat)

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False, gsplat=True):
        if init:
            downsample_factor = self.config["dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["dataset"]["pcd_downsample"]
        point_size = self.config["dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["dataset"]:
            if self.config["dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)

        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        if gsplat:
            fused_color = torch.from_numpy(np.asarray(pcd.colors)).float().cuda()
            fused_color = fused_color.clamp(1e-6, 1.0 - 1e-6)
            fused_color = inverse_sigmoid(fused_color)
            # fused_color = inverse_sigmoid(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        else:
            fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]

        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        return fused_point_cloud, features, scales, rots, opacities

    def create_pcd_from_image_and_depth_render(self, cam, ext_points, rgb, depth, init=False, gsplat=True):
        if init:
            downsample_factor = self.config["dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["dataset"]["pcd_downsample"]
        point_size = self.config["dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["dataset"]:
            if self.config["dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)

        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()

        if gsplat:

            # fused_color = inverse_sigmoid(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
            fused_color = torch.from_numpy(np.asarray(pcd.colors)).float().cuda()
            fused_color = fused_color.clamp(1e-6, 1.0 - 1e-6)
            fused_color = inverse_sigmoid(fused_color)
        else:
            fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())

        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        P1 = np.asarray(pcd.points).shape[0]


        fused_point_cloud2 = torch.cat((torch.from_numpy(np.asarray(pcd.points)).float().cuda() , ext_points))

        # dist2 = (
        #     torch.clamp_min(
        #         distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
        #         0.0000001,
        #     )
        #     * point_size
        # )
        # scales = torch.log(torch.sqrt(dist2))[..., None]


        dist2 = (
            torch.clamp_min(
                distCUDA2(fused_point_cloud2),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2[:P1]))[..., None]

        
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        return fused_point_cloud, features, scales, rots, opacities

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities, kf_id
    ):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None, gsplat=True
    ):
        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap, gsplat=gsplat)
        )
        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id
        )

    def extend_from_pcd_seq_render(
        self, cam_info, ext_points, kf_id=-1, init=False, scale=2.0, depthmap=None, gsplat=True
    ):
        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_image_render(cam_info, ext_points, init, scale=scale, depthmap=depthmap, gsplat=gsplat)
        )
        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.depths_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    max_steps=self.max_steps+1000,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self,target_opacity=0.01):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * target_opacity)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.depths_accum = self.depths_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.depths_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # new_xyz_gradient_accum = torch.zeros((new_xyz.shape[0], 1), device="cuda")
        # new_depths_accum = torch.zeros((new_xyz.shape[0], 1), device="cuda")
        # new_denom = torch.zeros((new_xyz.shape[0], 1), device="cuda")
        # new_max_radii2D = torch.zeros((new_xyz.shape[0]), device="cuda")

        # self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, new_xyz_gradient_accum))  #only reset new gaussian features
        # self.depths_accum = torch.cat((self.depths_accum, new_depths_accum))
        # self.denom = torch.cat((self.denom, new_denom))
        # self.max_radii2D = torch.cat((self.max_radii2D, new_max_radii2D))

        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)#scale 标准差
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )#[N]
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask] #[NEW_N,3]
        new_features_dc = self._features_dc[selected_pts_mask] #[NEW_N,1,3]
        new_features_rest = self._features_rest[selected_pts_mask]#[NEW_N,(SH+1)**2-1,3]
        new_opacities = self._opacity[selected_pts_mask]#[NEW_N,3]
        new_scaling = self._scaling[selected_pts_mask]#[NEW_N,3]
        new_rotation = self._rotation[selected_pts_mask]#[NEW_N,4]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        self.xyz_gradient_accum[selected_pts_mask] = 0.0
        self.depths_accum[selected_pts_mask] = 0.0
        self.denom[selected_pts_mask] = 0
        self.max_radii2D[selected_pts_mask] = 0.
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size): #
        grads = self.xyz_gradient_accum / self.denom # 归一化梯度[N,1]
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = torch.logical_and((self.get_opacity < min_opacity).squeeze(), (self.unique_kfIDs != self.unique_kfIDs.max()).cuda())
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            #big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            with torch.no_grad():
                ratio_s = self.get_scaling.max(dim = 1).values/self.get_scaling.min(dim = 1).values
                big_points_rs = ratio_s > 40.0
            # prune_mask = torch.logical_or(
            #     torch.logical_or(prune_mask, big_points_vs), big_points_ws
            # )
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            prune_mask = torch.logical_or(prune_mask, big_points_rs)
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter): # add z
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )

        # self.depths_accum[update_filter] += torch.abs(
        #     viewspace_point_tensor.grad[update_filter, 2:3]
        # )

        self.denom[update_filter] += 1

        #torch.norm(viewspace_point_tensor.grad[update_filter, :3], dim=-1, keepdim=True)

    def densify_and_prune_final_opt(self, max_grad, min_opacity, extent, max_screen_size):
        ...


    def split_gaussians(self, split_mask, samps):
        from water_gaussian.cudalight._torch_impl import quat_to_rotmat  # 从四元数转换为旋转矩阵的函数
        """
        加随机偏移后复制

        Args:
            split_mask (torch.Tensor): 布尔掩码，指示需要拆分的高斯点。
            samps (int): 每个高斯点拆分成的样本数量。

        Returns:
            Dict[str, torch.Tensor]: 拆分后的高斯参数。
        """
        n_splits = split_mask.sum().item()  # 需要拆分的高斯点数量
        # CONSOLE.log(f"Splitting {split_mask.sum().item() / self.num_points} gaussians: {n_splits}/{self.num_points}")

        stds = self.get_scaling[split_mask].repeat(samps, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        scaled_samples = torch.normal(mean=means, std=stds)

        # 生成随机样本，用于拆分高斯点的位置
        # centered_samples = torch.randn((samps * n_splits, 3), device="cuda") # 生成一个坐标标准正态分布的的点云[0,1]
        # scaled_samples = (
        #         self.get_scaling[split_mask].repeat(samps, 1) * centered_samples
        # )  # 根据缩放参数调整样本[N,3]

        # rots = build_rotation(self._rotation[split_mask]).repeat(samps, 1, 1) # 将四元数转换为旋转矩阵 原版3Dgs实现

        quats = self.get_rotation[split_mask] / self.get_rotation[split_mask].norm(dim=-1, keepdim=True)  # 归一化四元数
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # 将四元数转换为旋转矩阵 cuda实现
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()  # 应用旋转
        new_means = rotated_samples + self.get_xyz[split_mask].repeat(samps, 1)  # 计算新的均值

        # 拆分颜色特征
        new_features_dc = self._features_dc[split_mask].repeat(samps, 1, 1)  # (N,1,3)
        new_features_rest = self._features_rest[split_mask].repeat(samps, 1, 1)  # (N,coeff-1,3)

        # 拆分透明度
        new_opacities = torch.logit(torch.clamp(self.get_opacity[split_mask].repeat(samps, 1),1e-4, (1-1e-4) ))#(N, 1)

        # # 拆分缩放参数，并减小缩放以避免过大
        # size_fac = 1.6
        # new_scales = self._scaling[split_mask] - np.log(size_fac)
        # new_scales = new_scales.repeat(samps, 1)
        # self.get_scaling[split_mask] = (self.get_scaling[split_mask]) / size_fac  # 更新原始缩放参数

        new_scales = self.scaling_inverse_activation(
            self.get_scaling[split_mask].repeat(samps, 1) / (0.8 * samps)
        )

        # 拆分四元数
        new_quats = self.get_rotation[split_mask].repeat(samps, 1)
        # 复制元数据
        new_kf_ids = self.unique_kfIDs[split_mask.cpu()].repeat(samps)
        new_n_obs = self.n_obs[split_mask.cpu()].repeat(samps)

        # 更新高斯参数：
        self.densification_postfix(
            new_means,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scales,
            new_quats,
            new_kf_ids=new_kf_ids,
            new_n_obs=new_n_obs
        )

    def cull_gaussians(self,
                       extra_cull_mask:Optional[torch.Tensor],
                       iteration,
                       stop_split_at,
                       cull_screen_size,
                       cull_scale_thresh,
                       cull_alpha_thresh,
                       cull_alpha_thresh_post,
                       refine_every,
                       reset_alpha_every,
                       stop_screen_size_at
                       ):
        """
        删除不透明度低于阈值或尺寸过大的高斯点。

        Args:
            extra_cull_mask (Optional[torch.Tensor]):
                额外的裁剪掩码，用于指示除了现有裁剪标准外还需要裁剪的高斯点。

        Returns:
            :param extra_cull_mask: 被裁剪高斯点的掩码。torch.Tensor
            :type iteration: 当前步数 int
        """
        n_bef = self.get_xyz.shape[0]  # 裁剪前的高斯点数量


        # 根据当前步数选择不同的不透明度阈值
        if iteration < stop_split_at:
            _cull_alpha_thresh = cull_alpha_thresh*0.8
        else:
            _cull_alpha_thresh = cull_alpha_thresh_post

        # 裁剪不透明度低于阈值的高斯点
        culls = (self.get_opacity < _cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()  # 低不透明度的数量
        toobigs_count = 0

        if extra_cull_mask is not None:
            # 如果有额外的裁剪掩码，合并裁剪条件
            culls = culls | extra_cull_mask

        if iteration > refine_every * reset_alpha_every: # 100*5
            # 在一定步数后，裁剪尺寸过大的高斯点
            # toobigs = (self.get_scaling.max(dim=-1).values > cull_scale_thresh).squeeze() # cull_scale_thresh写死在外面不太灵活
            toobigs = (self.get_scaling.max(dim=-1).values > self.get_scaling.max(dim=-1).values.mean()*cull_scale_thresh).squeeze()
            if iteration < stop_screen_size_at:
                # 如果未达到停止屏幕尺寸裁剪的步数，进一步裁剪屏幕尺寸过大的高斯点
                assert self.max_radii2D is not None
                toobigs = toobigs | (self.max_radii2D > cull_screen_size).squeeze()
            culls = culls | toobigs  # 合并裁剪条件
            toobigs_count = torch.sum(toobigs).item()  # 尺寸过大的数量

        self.prune_points(culls) #delet mask gaussian
        print("cull gaussian")
        return culls  # 返回被裁剪的高斯点掩码


    def dup_gaussians(self, dup_mask: torch.Tensor):
        "直接复制"
        # --- 尺寸对齐 ---
        if dup_mask.shape[0] < self._xyz.shape[0]:
            dup_mask = torch.cat([dup_mask,
                                  dup_mask.new_zeros(self._xyz.shape[0] - dup_mask.shape[0])])

        # 选出需要复制的旧点
        new_xyz = self._xyz[dup_mask]
        new_features_dc = self._features_dc[dup_mask]
        new_features_rest = self._features_rest[dup_mask]
        new_opacities = self._opacity[dup_mask]
        new_scaling = self._scaling[dup_mask]
        new_rotation = self._rotation[dup_mask]

        # 元数据也要复制
        new_kf_ids = self.unique_kfIDs[dup_mask.cpu()]
        new_n_obs = self.n_obs[dup_mask.cpu()]
        self.xyz_gradient_accum[dup_mask] = 0.0
        self.denom[dup_mask] = 0
        self.max_radii2D[dup_mask] = 0.
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_ids,
            new_n_obs=new_n_obs,
        )




    def split_gaussians_along_axis(self, split_mask : torch.Tensor, samps: int = 2):
        from water_gaussian.cudalight._torch_impl import quat_to_rotmat  # 从四元数转换为旋转矩阵的函数
        """
        沿主轴方向可控拆分高斯点，生成 samps 个子高斯:
        """
        # Step 1: 主轴方向
        scale_base = self.get_scaling[split_mask]  # [N,3]
        max_axis_idx = scale_base.argmax(dim=1, keepdim=True)  # [N,1]
        principal_dirs = torch.zeros_like(scale_base).scatter(1, max_axis_idx, 1.0)  # [N,3]

        # Step 2: repeat + rotate
        sign = (torch.arange(samps, device=scale_base.device) % 2 * 2 - 1).view(1, samps, 1)

        principal_dirs = (principal_dirs.unsqueeze(1) * sign).reshape(-1, 3)  # [N*s,3]
        stds = scale_base.repeat_interleave(samps, dim=0)  # [N*s,3]
        offset_magnitude = torch.randn((stds.shape[0], 1), device="cuda") * stds.mean(dim=1, keepdim=True) * 0.3 * stds.max(dim=1,keepdim=True).values

        quats = self.get_rotation[split_mask] / self.get_rotation[split_mask].norm(dim=-1, keepdim=True)  # [N,4]
        rots = quat_to_rotmat(quats.repeat_interleave(samps, dim=0))  # [N*s,3,3]
        # rots = build_rotation(self._rotation[split_mask]).repeat(samps, 1, 1)

        # 主轴方向先旋转后再偏移
        principal_dirs_rot = torch.bmm(rots, principal_dirs.unsqueeze(-1)).squeeze(-1)  # [N*s,3]
        offsets = principal_dirs_rot * offset_magnitude  # [N*s,3]

        # Step 3: 加回原始中心
        new_means = self.get_xyz[split_mask].repeat_interleave(samps, dim=0) + offsets

        # 拆分颜色特征
        new_features_dc = self._features_dc[split_mask].repeat_interleave(samps, dim=0)  # (N,1,3)
        new_features_rest = self._features_rest[split_mask].repeat_interleave(samps, dim=0)  # (N,coeff-1,3)

        # 拆分透明度
        new_opacities = torch.logit(1.0 - torch.sqrt(1-self.get_opacity[split_mask].repeat_interleave(samps, dim=0)))# new_a = 1-(1-a)^0.5 when samps = 2

        # # 拆分缩放,体积守恒
        # new_scales = self.scaling_inverse_activation(
        #     self.get_scaling[split_mask].repeat_interleave(samps, dim=0)* (1.0 - torch.abs(principal_dirs) + torch.abs(principal_dirs) / (samps**(1/3)))
        # )

        """---------------------------------------------------------------------------------------"""
        # 拆分缩放,长度守恒
        scales_base = self.get_scaling[split_mask].repeat_interleave(samps, dim=0)  # [N*s, 3]

        # 只更新主轴方向的 scale
        offset_along_axis = offsets * principal_dirs  # [N*s, 3], 其余维度为 0
        new_scales = scales_base - 2 * offset_along_axis

        # 防止负数（特别是如果 offset 太大）
        new_scales = torch.clamp(new_scales, min=1e-4)

        new_scales = self.scaling_inverse_activation(new_scales)
        """---------------------------------------------------------------------------------------"""

        # 拆分四元数
        new_quats = self.get_rotation[split_mask].repeat(samps, 1)
        # 复制元数据
        new_kf_ids = self.unique_kfIDs[split_mask.cpu()].repeat(samps)
        new_n_obs = self.n_obs[split_mask.cpu()].repeat(samps)

        # 更新高斯参数：
        self.densification_postfix(
            new_means,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scales,
            new_quats,
            new_kf_ids=new_kf_ids,
            new_n_obs=new_n_obs
        )
    def merge_gaussians(self, merge_mask: torch.Tensor):
        """
        merge gaussian use average position opacity and color
        Args:
            merge_mask (torch.Tensor): merge mask, a boolean tensor indicating which points to merge.

        Returns:
            None
        """
        # if the mask shape is different than the xyz shape, we need to pad it
        assert merge_mask.shape[0] == self._xyz.shape[0], "Merge mask must be similar with xyz shape"
        if merge_mask.sum() <= 1:
            return  # nothing to merge

        xyz = self._xyz[merge_mask]  # (N, 3)
        features_dc = self._features_dc[merge_mask] # [N, 1, 3]
        features_rest = self._features_rest[merge_mask] # [N, (SH+1)**2-1, 3]
        opacity = self._opacity[merge_mask] # N,1
        scaling = self._scaling[merge_mask] # N,3
        rotation = self._rotation[merge_mask] # N,4

        new_xyz = xyz.mean(dim=0,keepdim=True) # 1,3
        new_features_dc = features_dc.mean(dim=0,keepdim=True) # 1,1,3
        new_features_rest = features_rest.mean(dim=0,keepdim=True) # 1,(SH+1)**2-1,3
        new_opacities = torch.logit(torch.clamp(self.get_opacity[merge_mask].mean(dim=0, keepdim=True), 1e-4, 1-1e-4)) # 1,1
        new_scaling = scaling.mean(dim=0,keepdim=True) # 1,3
        new_rotation = torch.nn.functional.normalize(rotation.mean(dim=0,keepdim=True), dim=-1) # 1,4

        merge_indices = torch.nonzero(merge_mask, as_tuple=False).squeeze(1)
        ref_idx = merge_indices[0].item() # we chose the first point's kf_id as the new kf_id
        new_kf_ids = self.unique_kfIDs[ref_idx:ref_idx+1]
        new_n_obs = self.n_obs[ref_idx:ref_idx+1]

        self.prune_points(merge_mask)  # delete the original gaussian

        # update gaussian parameters

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_ids,
            new_n_obs=new_n_obs
        )



