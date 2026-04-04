from typing import Union

import torch
from torch import nn

from gaussian.utils.graphics_utils import getProjectionMatrix2, getWorld2View2, focal2fov
from gaussian.utils.slam_utils import image_gradient, image_gradient_mask
from mast3r_slam.lietorch_utils import as_SE3_s
import cv2
import numpy as np
from mast3r_slam.mast3r_utils import resize_img
from PIL import Image

def show_image(img, title=None):

    rgb_temp = img.numpy()
    rgb_show = (rgb_temp - rgb_temp.min()) / (rgb_temp.max() - rgb_temp.min() + 1e-8)
    rgb_show = np.clip(rgb_temp, 0, 1)  # 确保数值不越

    cv2.imshow(title, (rgb_show * 255).astype(np.uint8))
    cv2.waitKey(1)


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        T_init,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        water_mask=None,
        #original_image_raw=None,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        # self.T_WC = torch.eye(4, device=device)
        self.T_CW = torch.eye(4, device=device)
        self.T_WC = torch.eye(4, device=device)

        self.R = T_init[:3, :3]
        self.T = T_init[:3, 3]

        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.T_CW[:3, :3] = self.R
        self.T_CW[:3, 3] = self.T

        R_inv = self.R.T  # R_wc = R_cw^T
        T_inv = - R_inv @ self.T  # t_wc = -R_cw^T * t_cw

        self.T_WC[:3, :3] = R_inv
        self.T_WC[:3, 3] = T_inv

        # R_inv = self.R.T
        # T_inv = -self.R.T@self.T.unsqueeze(1)

        # self.T_WC[:3, :3] = R_inv
        # self.T_WC[:3, 3:4] = T_inv
        

        self.original_image = color
        self.depth = depth
        self.grad_mask = None
        #self.original_image_raw = original_image_raw

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width


        self.water_mask = water_mask


        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )
    
    @staticmethod
    def init_from_tracking(frame, projection_matrix, cur_K, W, H, C_conf, water_mask=None, tstamp=None):
 
        T_WC_se, s = as_SE3_s(frame.T_WC) #
        pose = T_WC_se[0].inv().matrix().cuda()
        depth = frame.X_canon[:,2].reshape(H,W)*s.item()  #plus sclae point cloud
        Ck = frame.get_average_conf()[:,0].reshape(H,W).cpu().numpy()
        # depth_origin = depth.cpu().numpy()
        # depth = depth_origin.copy()
        depth = depth.cpu().numpy()

        #depth_origin[Ck < C_conf] = 0.
        depth[water_mask.squeeze().numpy() | (Ck < C_conf)] = 0.
        #depth[water_mask.squeeze().numpy()] = 0.
        fx = cur_K[0,0].item();  fy = cur_K[1,1].item() ; cx = cur_K[0,2].item(); cy = cur_K[1,2].item()
        #show_image(frame.uimg)

        # import matplotlib.pyplot as plt

        # plt.figure(figsize=(18,6))
        # plt.subplot(1,3,1).imshow(frame.uimg.numpy())
        # plt.subplot(1,3,2).imshow(frame.water_mask.numpy(), cmap='gray')
        # plt.subplot(1,3,3).imshow(depth)
        # plt.show(block=True)  # `block=True` 会阻塞程序，直到手动关闭窗口


        # plt.figure(figsize=(24,6))
        # plt.subplot(1,4,1).imshow(frame.uimg.numpy())
        # plt.subplot(1,4,2).imshow(frame.water_mask.numpy(), cmap='gray')
        # plt.subplot(1,4,3).imshow(depth)
        # plt.subplot(1,4,4).imshow(depth_origin)
        # plt.show(block=True)  # `block=True` 会阻塞程序，直到手动关闭窗口

        cam = Camera(
            frame.frame_id,
            frame.uimg.permute(2,0,1).cuda(),
            depth,
            pose,
            pose,
            projection_matrix,
            fx,
            fy,
            cx,
            cy,
            focal2fov(fx, W),
            focal2fov(fy, H),
            H,
            W,
            water_mask=water_mask.cuda()
            #original_image_raw=frame.original_image
            )
        
        cam.tstamp = tstamp
        return cam


    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, T ,  projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

        self.T_CW[:3, :3] = self.R
        self.T_CW[:3, 3] = self.T

        R_inv = self.R.T
        T_inv = -self.R.T@self.T

        self.T_WC[:3, :3] = R_inv
        self.T_WC[:3, 3] = T_inv

    def update_T(self, T):
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        
        self.T_CW[:3, :3] = self.R
        self.T_CW[:3, 3] = self.T

        R_inv = self.R.T
        T_inv = -self.R.T@self.T

        self.T_WC[:3, :3] = R_inv
        self.T_WC[:3, 3] = T_inv

    def compute_grad_mask(self, config):
        edge_threshold = config["training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if config["dataset"]["type"] == "replica":
            row, col = 32, 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            for r in range(row):
                for c in range(col):
                    block = img_grad_intensity[
                        :,
                        r * int(h / row) : (r + 1) * int(h / row),
                        c * int(w / col) : (c + 1) * int(w / col),
                    ]
                    th_median = block.median()
                    block[block > (th_median * multiplier)] = 1
                    block[block <= (th_median * multiplier)] = 0
            self.grad_mask = img_grad_intensity
        else:
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None
        #self.original_image_raw = None

        self.exposure_a = None
        self.exposure_b = None

    
    def resize_image(self, new_width, orig_image, origin_K, segmodel, cur_text_prompt):
        """
        Resize the camera's image resolution and update intrinsic parameters accordingly.

        Args:
            new_width: The new width for the image.
            orig_image: The original image tensor to resize.
        """
       
        # Resize the image
        if new_width == orig_image.shape[1]:
            self.fx = origin_K[0, 0]
            self.fy = origin_K[1, 1]
            self.cx = origin_K[0, 2]
            self.cy = origin_K[1, 2]
            self.image_height = orig_image.shape[0]
            self.image_width = orig_image.shape[1]
            self.FoVx = focal2fov(self.fx, self.image_width)
            self.FoVy = focal2fov(self.fy, self.image_height)
            
            gt_image = torch.from_numpy(orig_image)
            
            self.original_image = gt_image.permute(2,0,1).cuda()

            results = segmodel.predict_mask(Image.fromarray(np.uint8(orig_image * 255)), cur_text_prompt)
            #water_mask = keep_largest_component((1-results[0]))
            self.water_mask = results[...,None].bool()

            self.projection_matrix = getProjectionMatrix2(
                znear=0.01,
                zfar=100.0,
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
                W=self.image_width,
                H=self.image_height,
            ).transpose(0, 1).to(self.device)


        else:
            #resized_image = resize_img(orig_image, new_width, return_transformation=True)
            
            resized_image, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
                orig_image, new_width, return_transformation=True
            )

            self.fx = origin_K[0, 0] / scale_w
            self.fy = origin_K[1, 1] / scale_h
            self.cx = origin_K[0, 2] / scale_w - half_crop_w
            self.cy = origin_K[1, 2] / scale_h - half_crop_h

            cur_origin_img = resized_image["unnormalized_img"]
            self.image_height = cur_origin_img.shape[0]
            self.image_width = cur_origin_img.shape[1]

            self.FoVx = focal2fov(self.fx, self.image_width)
            self.FoVy = focal2fov(self.fy, self.image_height)
            
            gt_image = torch.from_numpy(cur_origin_img) / 255.0
            
            self.original_image = gt_image = gt_image.permute(2,0,1).cuda()

            results = segmodel.predict_mask(Image.fromarray(cur_origin_img), cur_text_prompt)
            #water_mask = keep_largest_component((1-results[0]))
            self.water_mask = results[...,None].bool()


            self.projection_matrix = getProjectionMatrix2(
                znear=0.01,
                zfar=100.0,
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
                W=self.image_width,
                H=self.image_height,
            ).transpose(0, 1).to(self.device)

            

    def rescale_output_resolution(
            self,
            scaling_factor: Union[float, int, torch.Tensor],
            scale_rounding_mode: str = "floor",
    ) -> None:
        """
        Rescale the camera's intrinsic parameters and image resolution.

        Args:
            scaling_factor: A float, int, or tensor specifying the scaling factor.
            scale_rounding_mode: Method to round the scaled height and width. Options are 'floor', 'round', or 'ceil'.
        """
        # Convert to tensor if necessary
        if isinstance(scaling_factor, (float, int)):
            scaling_factor = torch.tensor(scaling_factor, dtype=torch.float32, device=self.device)
        elif isinstance(scaling_factor, torch.Tensor):
            scaling_factor = scaling_factor.to(self.device).float()
            if scaling_factor.numel() != 1:
                raise ValueError("Only scalar scaling_factor is supported in Camera class.")
            scaling_factor = scaling_factor.squeeze()
        else:
            raise TypeError("scaling_factor must be a float, int, or 0-dim torch.Tensor.")

        # Scale intrinsics
        self.fx *= scaling_factor
        self.fy *= scaling_factor
        self.cx *= scaling_factor
        self.cy *= scaling_factor

        # Scale resolution
        height = self.image_height * scaling_factor
        width = self.image_width * scaling_factor

        if scale_rounding_mode == "floor":
            self.image_height = int(torch.floor(height).item())
            self.image_width = int(torch.floor(width).item())
        elif scale_rounding_mode == "round":
            self.image_height = int(torch.round(height).item())
            self.image_width = int(torch.round(width).item())
        elif scale_rounding_mode == "ceil":
            self.image_height = int(torch.ceil(height).item())
            self.image_width = int(torch.ceil(width).item())
        else:
            raise ValueError("Invalid scale_rounding_mode. Choose from 'floor', 'round', or 'ceil'.")

        # Optional: Recompute FoV
        self.FoVx = focal2fov(self.fx, self.image_width)
        self.FoVy = focal2fov(self.fy, self.image_height)

        # Optional: Update projection matrix if needed
        self.projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            W=self.image_width,
            H=self.image_height,
        ).transpose(0, 1).to(self.device)


class Nonkeyframe_Camera(nn.Module):
    def __init__(
        self,
        uid,
        T_delta,
        T_k,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Nonkeyframe_Camera, self).__init__()
        self.uid = uid   #frame id
        self.device = device

        #self.T_WC = torch.eye(4, device=device)
        self.T_delta = T_delta  #T_fk const
        self.T_k = T_k  #T_kw


        self.T_CW = self.T_delta@self.T_k if self.T_k is not None else None


        self.R = self.T_CW[:3, :3] if self.T_CW is not None else None
        self.T = self.T_CW[:3, 3] if self.T_CW is not None else None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)


    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, T ,  projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

        self.T_CW[:3, :3] = self.R
        self.T_CW[:3, 3] = self.T


    def update_Tk(self, T_k):

        self.T_k = T_k
        self.T_CW = self.T_delta@self.T_k

        self.R = self.T_CW[:3, :3]
        self.T = self.T_CW[:3, 3]


    def update_T(self, T):
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        
        self.T_CW[:3, :3] = self.R
        self.T_CW[:3, 3] = self.T

    def to_original_size(self, original_height, original_width, K_orig):
        """
        Rescales the camera parameters to the original image size.

        Args:
            original_height (int): The original height of the image.
            original_width (int): The original width of the image.
            K_orig (torch.Tensor): The original camera intrinsic matrix (3x3).
        """
        if self.image_height == original_height and self.image_width == original_width:
            return

        # Calculate scaling factors
        # Update intrinsic parameters
        self.fx = K_orig[0, 0].item()
        self.fy = K_orig[1, 1].item()
        self.cx = K_orig[0, 2].item()
        self.cy = K_orig[1, 2].item()
        
        # Update image dimensions
        self.image_height = original_height
        self.image_width = original_width

        # Recalculate FoV
        self.FoVx = focal2fov(self.fx, self.image_width)
        self.FoVy = focal2fov(self.fy, self.image_height)

        # Update projection matrix
        self.projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            W=self.image_width,
            H=self.image_height,
        ).transpose(0, 1).to(self.device)