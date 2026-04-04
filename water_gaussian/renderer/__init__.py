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

import math

import torch
import torch.nn.functional as F
from water_gaussian.rendering import rasterization_inria_wrapper, rasterization, _rasterization
from water_gaussian.rasterizerlight.project_gaussians import project_gaussians
from water_gaussian.rasterizerlight.rasterize import rasterize_gaussians

from gaussian.scene.gaussian_model_water import GaussianModel
from gaussian.utils.sh_utils import eval_sh
from mast3r_slam.network import positional_encode_directions, SHEncoding, Medium, Light, BRDF
from water_gaussian.utils import spherical_harmonics




def render(viewpoint_camera, pc: GaussianModel,
           #pipe,
           scaling_modifier=1.0,
           antialiasing=True,
           Medium_MLP=None,
           Light_MLP=None,
           BRDF_mlp = None,
           config_sh_degree_interval=1000,
           iteration=-1,
           shrefine = False,
           device = "cuda",
           clip_thresh=0.01
           ):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if pc.get_xyz.shape[0] == 0:
        return None

    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=device
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass


    viewmat = torch.eye(4, device=device, dtype=pc.get_xyz.dtype)
    viewmat[:3, :3] = viewpoint_camera.R
    viewmat[:3, 3] = viewpoint_camera.T # T_cw

    cx = viewpoint_camera.cx
    cy = viewpoint_camera.cy
    H = viewpoint_camera.image_height
    W = viewpoint_camera.image_width

    y = torch.linspace(0., H, H, device=device)
    x = torch.linspace(0., W, W, device=device)

    yy, xx = torch.meshgrid(y, x)
    yy = (yy - viewpoint_camera.cy) / viewpoint_camera.fy
    xx = (xx - viewpoint_camera.cx) / viewpoint_camera.fx

    directions = torch.stack([yy, xx, -1 * torch.ones_like(xx)], dim=-1) # directions_flat must be normed
    norms = torch.linalg.norm(directions, dim=-1, keepdim=True)
    directions = directions / norms
    directions = directions @ viewpoint_camera.R.float() # [h,w,3]
    directions_flat = directions.view(-1, 3)  # [H*W, 3]
    # directions_encoded = positional_encode_directions(directions_flat, L=pc.L_degree) # [H*W, 24]

    encoder = SHEncoding(levels=4, implementation="tcnn")  # SHEncoding 16-dim output
    directions_encoded = encoder(directions_flat)  # [H*W, 16]
    outputs_shape = directions.shape[:-1] # [H,W]

    # Medium MLP forward pass

    #Medium_MLP = Medium(levels=4, hidden_dim=128, num_layers=4, mlp_type="tcnn").to("cuda")#

    # Light_MLP = Light(levels=4, hidden_dim=128, num_layers=4, mlp_type="tcnn").to("cuda")

    # BRDF_MLP= BRDF(hidden_dim=128, num_layers=4, implementation="torch").to("cuda")
    #"_____________brdf_______________"
    # view_dir = directions_flat  # [H*W, 3]
    # light_dir = directions_flat  # can change to light vector [H*W, 3]
    # normal = directions_flat
    # roughness = torch.full((view_dir.shape[0], 1), 0.3, device=view_dir.device)  # 示例值
    # specular = torch.full((view_dir.shape[0], 3), 0.04, device=view_dir.device)  # 默认 F0

    if Medium_MLP is None:
        medium_base_out = pc.Medium_MLP(directions_encoded)
        colour_activation = pc.Medium_MLP.colour_activation
        sigma_activation = pc.Medium_MLP.sigma_activation
        density_bias = pc.Medium_MLP.density_bias
    else:
        medium_base_out = Medium_MLP(directions_encoded)
        colour_activation = Medium_MLP.colour_activation
        sigma_activation = Medium_MLP.sigma_activation
        density_bias = Medium_MLP.density_bias

    # if Light_MLP is None:
    #     light_base_out = pc.Light_MLP(directions_encoded)
    # else:
    #     light_base_out = Light_MLP(directions_encoded)
    # if Light_MLP is None:
    #     light_base_out = pc.Light_MLP(directions_encoded)
    # else:
    #     light_base_out = Light_MLP(directions_encoded)

    # if BRDF_MLP is None:
    #     rgb_brdf = pc.BRDF_model(view_dir=view_dir,normal=normal,light_dir=light_dir,roughness=roughness,specular=specular)  # [H*W, 3]
    # else:
    #     rgb_brdf = BRDF_MLP(view_dir=directions_flat, normal=directions_flat, light_dir=directions_flat, roughness=roughness,specular=specular)  # [H*W, 3]

    medium_rgb = colour_activation(medium_base_out[..., :3]).view(*outputs_shape, -1)
    medium_bs = sigma_activation(medium_base_out[..., 3:6] + density_bias).view(*outputs_shape, -1)
    medium_attn = sigma_activation(medium_base_out[..., 6:] + density_bias).view(*outputs_shape, -1)


    # enhanced_color_rgb = light_base_out.view(*outputs_shape , -1) # /enhance phi
    # enhanced_color_rgb = rgb_brdf.view(*outputs_shape , -1)
    # medium_rgb = torch.zeros(*outputs_shape,3).to("cuda") # test
    # medium_bs = torch.ones(*outputs_shape,3).to("cuda")
    # medium_attn = torch.zeros(*outputs_shape,3).to("cuda")
    enhanced_color_rgb = torch.ones(*outputs_shape,3).to("cuda")

    # color_rgb = torch.ones(*outputs_shape,3).to("cuda") # /phi

    colors_crop = pc.get_features
    BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

    means3D = pc.get_xyz # global
    scales = pc.get_scaling * scaling_modifier
    quats = pc.get_rotation # quats must be normed




    # means_crop = means3D #we dont need crop mean
    # scales_crop = scales # scales
    # quats_crop = rotations # rotation /norm quat


    xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
        means3D,
        scales,
        1,
        quats, #
        viewmat.squeeze()[:3, :],
        viewpoint_camera.cam_rot_delta,
        viewpoint_camera.cam_trans_delta,
        viewpoint_camera.fx,
        viewpoint_camera.fy,
        cx,
        cy,
        H,
        W,
        BLOCK_WIDTH,
        clip_thresh=clip_thresh,
    )  # type: ignore

    #if pc.max_sh_degree > 0 and shrefine == True:

    if pc.active_sh_degree > 0:
        eps = 1e-9
        cameraInWorld = viewpoint_camera.T_WC

        # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)  # 将SH特征的形状调整为（batch_size * num_points，3，(max_sh_degree+1)**2）。
        # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0],1))  # 计算相机中心到每个点的方向向量，并归一化。
        # dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + eps) # 计算相机中心到每个点的方向向量，并归一化。
        # colors1 = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) #使用SH特征将方向向量转换为RGB颜色。
        # colors1 = torch.clamp_min(colors1 + 0.5, 0.0)  # 将RGB颜色的范围限制在0到1之间。
        # if (colors1<0).any() :
        #     raise AssertionError (f"colors1<0   {iteration}    ")
        # if (colors1>1).any() :
        #     raise AssertionError (f"colors1>1,{iteration}")
        # assert torch.isfinite(colors1).all(), f"colors1 has NaN or Inf at step {iteration}"

        viewdirs = means3D.detach() - cameraInWorld[:3, 3].unsqueeze(0).detach()  # (n, 3)
        viewdirs = viewdirs / (viewdirs.norm(dim=-1, keepdim=True)+eps)
        n = pc.active_sh_degree
        colors = spherical_harmonics(n, viewdirs, colors_crop)
        colors = torch.clamp(colors + 0.5, min=0.0)
        # if (colors<0).any() :
        #     raise AssertionError (f"colors<0,{iteration}")
        # if (colors>1).any() :
        #     raise AssertionError (f"colors>1,{iteration}")
        # assert torch.isfinite(colors).all(), f"colors has NaN or Inf at step {iteration}"
        # if n>=2:
        #     print(pc._features_rest)

    else:
        colors = torch.sigmoid(colors_crop[:,0,:])  # [N, K, 3]

    if antialiasing:
        input_opacity = pc.get_opacity*comp[..., None]
    else:
        input_opacity = pc.get_opacity

    xys_grad_abs = torch.zeros_like(xys)  # 每一次训练初始化为0

    rgb_object, rgb_clear, rgb_medium, depth_im, alpha = rasterize_gaussians(
        xys,
        xys_grad_abs,
        depths,
        radii,
        conics,
        num_tiles_hit,
        colors,
        input_opacity,
        medium_rgb,
        medium_bs,
        medium_attn,
        enhanced_color_rgb,
        H,
        W,
        BLOCK_WIDTH,
        background=medium_rgb,
        return_alpha=True,
    )

    rgb_object = torch.clamp(rgb_object, 0, 1)
    rgb = rgb_object + rgb_medium  # rgb_object[0-1],rgb_medium[0-1]
    rgb = torch.clamp(rgb, 0, 1)
    # rgb_clear = torch.clamp(rgb_clear, 0., 1.)  # [R,G,B]
    depth_im = depth_im[..., None]
    alpha = alpha[..., None]
    depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())

    # imshow(rgb, "rgb")
    # imshow(rgb_object, "rgb_object")
    # imshow(rgb_clear, "rgb_clear")
    # imshow(viewpoint_camera.original_image.permute(1,2,0), "original_image")

    #For gs slam render out [H,W,C] -> [C,H,W]
    rgb = rgb.permute(2, 0, 1)
    rgb_clear = rgb_clear.permute(2, 0, 1)
    depth_im = depth_im.permute(2, 0, 1)
    rgb_object = rgb_object.permute(2, 0, 1)
    alpha =alpha.permute(2, 0, 1)
    rgb_medium = rgb_medium.permute(2,0,1)
    medium_bs = medium_bs.permute(2,0,1)
    medium_attn = medium_attn.permute(2,0,1)

    try:
        xys.retain_grad()
    except Exception:
        pass

    return {"render": rgb,
            "viewspace_points":xys,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth_im,
            "accumulation": alpha, "background": medium_rgb,
            "rgb_object": rgb_object, "rgb_clear": rgb_clear, "rgb_medium": rgb_medium,
            "medium_bs": medium_bs, "medium_attn": medium_attn,
            "opacity": alpha}

    # rendered_image = render_colors[0].permute(2, 0, 1)
    # radii = info["radii"].squeeze(0)  # [N,]
    # try:
    #     info["means2d"].retain_grad()  # [1, N, 2]
    # except:
    #     pass
    #
    # # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # # They will be excluded from value updates used in the splitting criteria.
    # return {"render": rendered_image,
    #         "viewspace_points": info["means2d"],
    #         "visibility_filter": radii > 0,
    #         "radii": radii}

def imshow(image,title):
    import cv2
    import numpy as np
    rgb_temp = image.detach().cpu().numpy()
    
    rgb_temp = rgb_temp[:,:,::-1]
    rgb_show = np.clip(rgb_temp, 0, 1)  # 确保数值不越

    cv2.imshow(title, (rgb_show * 255).astype(np.uint8))
    cv2.waitKey(1)
    ...




if __name__ == "__main__":
        import matplotlib

        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.title('Test Plot')
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
