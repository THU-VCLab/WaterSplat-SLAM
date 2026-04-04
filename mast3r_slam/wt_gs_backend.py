import random
import time

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from matplotlib.scale import get_scale_names
from tqdm import trange
from munch import munchify
from lietorch import SE3, SO3
import matplotlib.pyplot as plt

from gaussian.utils.logging_utils import Log, clone_obj
from water_gaussian.renderer import render # use modified gsplat rasterizer
#from gaussian.renderer import render
from gaussian.utils.loss_utils import l1_loss, l1_loss_mask, ssim
from gaussian.scene.gaussian_model_water import GaussianModel
from gaussian.utils.graphics_utils import getProjectionMatrix2, focal2fov
from gaussian.utils.pose_utils import update_pose
from gaussian.utils.slam_utils import to_se3_vec, get_loss_mapping_rgbd, get_loss_mapping_rgbd_mask, get_loss_mapping_rgb, get_loss_mapping_rgb_mask, get_loss_mapping_all,get_water_loss, get_water_loss_mask
from gaussian.utils.camera_utils import Camera, Nonkeyframe_Camera
from gaussian.utils.eval_utils import eval_rendering, eval_rendering_kf, eval_ate, save_gaussians, save_render_keyframes, save_render_noneframes
from gui_water import gui_utils, slam_gui
from mast3r_slam.network import MLP, positional_encode_directions
from mast3r_slam.lietorch_utils import as_SE3_s_cuda, as_SO3_cuda, as_SE3
import lietorch
from tqdm import tqdm
from mast3r_slam.mast3r_utils import resize_img
from PIL import Image
from collections import defaultdict
from mast3r_slam.generate_voxel import Voxel

# from https://github.com/Willyzw/HI-SLAM2/blob/main/hislam2/gs_backend.py
class WTGSBackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.iteration_count = 0
        self.viewpoints = {}
        self.current_window = []
        self.initialized = False
        self.save_dir = config["output"]["gs_dir"]
        self.use_gui = config["results"]["use_gui"]

        self.opt_params = munchify(config["opt_params"])
        self.config["training"]["monocular"] = False

        self.embedded_dim = self.config["training"]["embedded_dim"]
        self.L_degree = self.config["training"]["L_degree"]
        self.layer_num = self.config["training"]["layer_num"]
        self.gsplat_enable = self.config["training"]["gsplat"]

        self.gaussians = GaussianModel(sh_degree=self.config["color_refine"]["sh_max"], embedded_dim=self.embedded_dim, L_degree=self.L_degree, layer_num=self.layer_num, config=self.config)
        
        
        self.rot_thre = self.config["training"]["rot_thre"]
        self.trans_thre = self.config["training"]["trans_thre"]
        self.normal_mapping_cur_frame = self.config["training"]["normal_mapping_cur_frame"]
        self.normal_mapping_past_frame = self.config["training"]["normal_mapping_past_frame"]
        self.normal_mapping_iters = self.config["training"]["normal_mapping_iters"]

        self.frame_up_ratio = self.config["training"]["frame_up_ratio"]
        self.opacity_thre = self.config["training"]["opacity_thre"]

        self.glomap_outiters = self.config["training"]["glomap_outiters"]
        self.glomap_initers = self.config["training"]["glomap_initers"]
        self.glomap_window = self.config["training"]["glomap_window"]

        self.xyz_gradient_attnratio = self.config["training"]["xyz_gradient_attnratio"]
        self.refinement_xyz_gradient_attnratio = self.config["training"]["refinement_xyz_gradient_attnratio"]

        self.new_mapping_iter = self.config["training"]["new_mapping_iter"]

        self.kf_overlap = self.config["training"]["kf_overlap"]
        self.use_kf_BA = self.config["training"]["use_kf_BA"]

        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        self.cameras_extent = 6.0
        self.projection_matrix = None

        self.Nonkeyframe_viewpoints = dict()
        self.all_nonkf_frames = None

        self.text_prompt = self.config["training"]["text_prompt"]
        self.datasets = None
        self.segmodel = None
        
        self.kf_ids = []
        self.nonid_2kfid = dict()

        self.isotropic_losses = []

        self.antialiasing = self.config["training"]["antialiasing"]

        self.GSBA_iter_perframe = self.config["training"]["GSBA_iter_perframe"]
        self.GSBA_window = self.config["training"]["GSBA_window"]
        self.use_depth_loss = self.config["training"]["use_depth_loss"]
        self.use_iso_loss = self.config["training"]["use_iso_loss"]

        "--------------color_refine---------------"
        self.clip_thresh = self.config["color_refine"]["clip_thresh"]
        "--------------pyramid_config---------------"
        self.use_pyramid:bool = config["training"]["use_pyramid"]
        self.pyramid_num = self.config["training"]["pyramid_num"]
        self.origin_width = None
        self.origin_height = None
        self.origin_K = None

        # ---- gsplat params ----
        self.config_sh_degree_interval = self.config["color_refine"]["config_sh_degree_interval"]
        """SH degree upgrade interval (iterations)"""
        self.refine_every: int = config["color_refine"]["refine_every"]
        """Refinement interval: cull and densify every N steps"""
        self.cull_alpha_thresh: float = config["color_refine"]["cull_alpha_thresh"]
        """Alpha threshold for culling during densification"""
        self.cull_alpha_thresh_post: float = config["color_refine"]["cull_alpha_thresh_post"]
        """Alpha threshold for culling after densification"""
        self.reset_alpha_thresh: float = config["color_refine"]["reset_alpha_thresh"]
        """Target value when resetting opacity"""
        self.cull_scale_thresh: float = config["color_refine"]["cull_scale_thresh"]
        """3D scale threshold for culling oversized Gaussians"""
        self.continue_cull_post_densification: bool = True
        """If True, continue culling after densification stops"""
        self.reset_alpha_every: int = config["color_refine"]["reset_alpha_every"]
        """Reset opacity every N refine cycles (actual interval = reset_alpha_every * refine_every)"""
        self.densify_grad_thresh: float = config["color_refine"]["densify_grad_thresh"]
        """Gradient norm threshold for densification"""
        self.densify_size_thresh: float = config["color_refine"]["densify_size_thresh"]
        """3D scale threshold: split if above, clone if below"""
        self.n_split_samples: int = config["color_refine"]["n_split_samples"]
        """Number of samples when splitting a Gaussian"""
        self.clip_thresh: float = config["color_refine"]["clip_thresh"]
        """Minimum depth clipping threshold for rendering"""
        self.cull_screen_size: float = config["color_refine"]["cull_screen_size"]
        """Cull Gaussians whose 2D projection exceeds this size (pixels, unnormalized)"""
        self.split_screen_size: float = config["color_refine"]["split_screen_size"]
        """Split Gaussians whose 2D projection exceeds this ratio"""
        self.stop_screen_size_at: int = config["color_refine"]["stop_screen_size_at"]
        """Stop screen-size-based culling/splitting after this iteration"""
        self.stop_split_at: int = config["color_refine"]["stop_split_at"]
        """Stop splitting Gaussians after this iteration"""
        self.max_gaussian_points: int = config["color_refine"]["max_gaussian_points"]
        """Max Gaussian count; only cull (no densify) when exceeded"""
        "--------------------debug config---------------"
        self.debug = config["debug_config"]["debug"]
        self.debug_showImage = config["debug_config"]["debug_showImage"]
        self.debug_showid = config["debug_config"]["debug_showid"]


        self.set_hyperparams()



        if self.use_gui:
            self.q_main2vis = mp.Queue()
            self.q_vis2main = mp.Queue()
            self.params_gui = gui_utils.ParamsGUI(
                background=self.background,
                gaussians=self.gaussians,
                q_main2vis=self.q_main2vis,
                q_vis2main=self.q_vis2main,
            )
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui, self.embedded_dim, self.L_degree, self.layer_num))
            gui_process.start()
            time.sleep(3)

    def set_hyperparams(self):
        self.init_itr_num = self.config["training"]["init_itr_num"]
        self.init_gaussian_update = self.config["training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["training"]["init_gaussian_th"]
        self.init_gaussian_extent = self.cameras_extent * self.config["training"]["init_gaussian_extent"]
        self.gaussian_update_every = self.config["training"]["gaussian_update_every"]
        self.global_gaussian_update_every = self.config["training"]["global_gaussian_update_every"]
        self.refinement_gaussian_update_every = self.config["color_refine"]["gaussian_update_every"]
        self.refinement_gaussian_update_offset = self.config["color_refine"]["gaussian_update_offset"]
        self.gaussian_update_offset = self.config["training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["training"]["gaussian_th"]
        self.gaussian_th_refine = self.config["color_refine"]["gaussian_th_refine"]
        self.gaussian_extent = self.cameras_extent * self.config["training"]["gaussian_extent"]
        self.gaussian_extent_refine = self.cameras_extent * self.config["color_refine"]["gaussian_extent_refine"]
        self.gaussian_reset = self.config["training"]["gaussian_reset"]
        self.size_threshold = self.config["training"]["size_threshold"]
        self.size_threshold_refine = self.config["color_refine"]["size_threshold_refine"]
        self.window_size = self.config["training"]["window_size"]
        self.lambda_dnormal = self.config["training"]["lambda_dnormal"]

    def init_projection_matrix(self, cur_K, W, H):
        
        self.projection_matrix = getProjectionMatrix2(znear=0.01, zfar=100.0, fx = cur_K[0,0].item(), \
            fy = cur_K[1,1].item() , cx = cur_K[0,2].item(), cy = cur_K[1,2].item(), W=W, H=H).transpose(0, 1).cuda()
        
    def add_new_nonkf(self, non_kf_viewpoint, last_keyid):

        if last_keyid not in self.Nonkeyframe_viewpoints.keys():
            self.Nonkeyframe_viewpoints[last_keyid] = dict()
        self.Nonkeyframe_viewpoints[last_keyid][non_kf_viewpoint.uid] = non_kf_viewpoint

        self.nonid_2kfid[non_kf_viewpoint.uid] = last_keyid

    def update_nonkf_pose(self):

        for kf_id in self.Nonkeyframe_viewpoints.keys():
            key_pose = self.viewpoints[kf_id].T_CW
            for non_kf_viewpoint in self.Nonkeyframe_viewpoints[kf_id].values():
                non_kf_viewpoint.update_Tk(key_pose)
                
    def get_all_nonkf(self):

        self.all_nonkf_frames = []
        for kf_id in self.Nonkeyframe_viewpoints.keys():
            for non_kf_viewpoint in self.Nonkeyframe_viewpoints[kf_id].values():
                #non_kf_viewpoint.to_original_size(self.origin_height, self.origin_width, self.origin_K)
                self.all_nonkf_frames.append(non_kf_viewpoint)

    def get_all_viewpoints(self):

        self.all_viewpoints = dict()
        for kf_id in self.kf_ids:
            self.all_viewpoints[kf_id] = self.viewpoints[kf_id]
            if kf_id in self.Nonkeyframe_viewpoints.keys():
                for nonkf_id in sorted(self.Nonkeyframe_viewpoints[kf_id]):
                    non_kf_viewpoint = self.Nonkeyframe_viewpoints[kf_id][nonkf_id]
                    self.all_viewpoints[nonkf_id] = non_kf_viewpoint

    def compute_intersections(self, viewpoint1, viewpoint2):

        with torch.no_grad():
            render_pkg1 = render(viewpoint1, self.gaussians, antialiasing=self.antialiasing)
            vis1 = render_pkg1["visibility_filter"]

            render_pkg2 = render(viewpoint2, self.gaussians, antialiasing=self.antialiasing)
            vis2 = render_pkg2["visibility_filter"]


            # image1 = render_pkg1["render"].cpu().numpy().transpose(1, 2, 0)
            # image2 = render_pkg2["render"].cpu().numpy().transpose(1, 2, 0)

            # gt_image1 = viewpoint1.original_image.cpu().numpy().transpose(1, 2, 0)
            # gt_image2 = viewpoint2.original_image.cpu().numpy().transpose(1, 2, 0)


            # import matplotlib.pyplot as plt

            # plt.figure(figsize=(24,24))
            # plt.subplot(2,2,1).imshow(image1)
            # plt.title("render image_" + str(viewpoint1.uid))
            # plt.subplot(2,2,2).imshow(image2)
            # plt.title("render image_" + str(viewpoint2.uid))

            # plt.subplot(2,2,3).imshow(gt_image1)
            # plt.title("gt image_" + str(viewpoint1.uid))
            # plt.subplot(2,2,4).imshow(gt_image2)
            # plt.title("gt image2_" + str(viewpoint2.uid))
            # #plt.subplot(1,3,3).imshow(depth)
            # plt.show(block=True)  # block=True blocks until window is closed




            union = torch.logical_or(
                vis1, vis2
                ).count_nonzero()
            intersection = torch.logical_and(
                vis1, vis2
                ).count_nonzero()
            point_ratio = intersection / union

        return point_ratio
    
    def convert_non2kf(self, kf_id, nonkf_viewpoint: Nonkeyframe_Camera, img_size=512):

       
        key_pose = self.viewpoints[kf_id].T_CW
        nonkf_viewpoint.update_Tk(key_pose)


        _, gt_image = self.datasets[nonkf_viewpoint.uid]

        uimg = resize_img(gt_image, img_size)["unnormalized_img"]
        gt_image = torch.from_numpy(uimg) / 255.0
        gt_image = gt_image.permute(2,0,1).cuda()

        uid = nonkf_viewpoint.uid ; color = gt_image; depth = None; T_init = nonkf_viewpoint.T_CW; gt_T =  nonkf_viewpoint.T_CW
        projection_matrix = nonkf_viewpoint.projection_matrix
        fx = nonkf_viewpoint.fx ; fy = nonkf_viewpoint.fy ; cx = nonkf_viewpoint.cx ; cy = nonkf_viewpoint.cy
        fovx = nonkf_viewpoint.FoVx ; fovy = nonkf_viewpoint.FoVy
        image_height = nonkf_viewpoint.image_height ; image_witdh = nonkf_viewpoint.image_width

        cur_text_prompt = self.text_prompt
        results = self.segmodel.predict_mask(Image.fromarray(uimg), cur_text_prompt)
        #water_mask = keep_largest_component((1-results[0]))
        water_mask = results[...,None].bool()

        new_viewpoint = Camera(uid, color, depth, T_init, gt_T, projection_matrix, fx, fy, cx, cy, fovx, fovy, image_height, image_witdh, water_mask=water_mask)
        
        return new_viewpoint
        
    def insert_kf(self, kf_id1, kf_id2):
        
        if kf_id1 == kf_id2 -1:
            return
        
        kf_viewpoint1 = self.all_viewpoints[kf_id1]
        kf_viewpoint2 = self.all_viewpoints[kf_id2]
        point_ratio = self.compute_intersections(kf_viewpoint1, kf_viewpoint2)
        if point_ratio > self.kf_overlap:
            return 
        mid_id = (kf_id1 + kf_id2)//2
        assert(mid_id not in self.kf_ids)

        related_kfid = self.nonid_2kfid[mid_id] 
        new_viewpoint = self.convert_non2kf(related_kfid, self.all_viewpoints[mid_id])

        

        #add new kf and delete midid nonkf
        self.viewpoints[mid_id] = new_viewpoint

        #with torch.no_grad():
            # render_pkg = render(new_viewpoint, self.gaussians, antialiasing=self.antialiasing)
            # gt_image = new_viewpoint.original_image.cpu().numpy().transpose(1, 2, 0)


            # import matplotlib.pyplot as plt

            # plt.figure(figsize=(12,6))
            # plt.subplot(1,2,1).imshow(render_pkg["render"].cpu().numpy().transpose(1, 2, 0))
            # plt.title("render image_" + str(new_viewpoint.uid))
            # plt.subplot(1,2,2).imshow(gt_image)
            # plt.title("render image_" + str(new_viewpoint.uid))

            # plt.show(block=True)  # block=True blocks until window is closed

        
        self.kf_ids.append(mid_id)
        self.all_viewpoints[mid_id] = new_viewpoint

        

        self.Nonkeyframe_viewpoints[related_kfid].pop(mid_id)

        self.insert_kf(kf_id1, mid_id)
        self.insert_kf(mid_id, kf_id2)

    def insert_kf_all(self):
        last_id = len(self.all_viewpoints)-1
        if last_id not in self.kf_ids:
            new_viewpoint = self.convert_non2kf(max(self.kf_ids), self.all_viewpoints[last_id])

            related_kfid = self.nonid_2kfid[last_id] 

            self.Nonkeyframe_viewpoints[related_kfid].pop(last_id)

            self.viewpoints[last_id] = new_viewpoint
            
            self.kf_ids.append(last_id)
            self.all_viewpoints[last_id] = new_viewpoint

           
            
        self.kf_ids = sorted(self.kf_ids)

        for i in range(len(self.kf_ids)-1):
            kf_id1 = self.kf_ids[i]
            kf_id2 = self.kf_ids[i+1]
            self.insert_kf(kf_id1, kf_id2)

    def process_track_data(self, keyframes, C_conf, gs_initialized, finalize=False):
        #find old pose and cur poses
        #get new s and new poses
        if not gs_initialized:
            cur_frame = keyframes[0]
            resize_H = cur_frame.img_shape[0,0].item()  ; resize_W = cur_frame.img_shape[0,1].item() 

            self.init_projection_matrix(cur_frame.K, resize_W, resize_H)
            viewpoint = Camera.init_from_tracking(cur_frame, self.projection_matrix, cur_frame.K, resize_W, resize_H, C_conf, cur_frame.water_mask, None)
            self.add_next_kf(0, viewpoint, init = True, depth_map = viewpoint.depth) 
            self.initialize_map(0, viewpoint)  # init map
            return
        
        len_keyframes = len(keyframes)
        new_poses_sim3 = lietorch.Sim3(torch.stack([keyframes[i].T_WC.data for i in range(len_keyframes)]))
        new_poses_se3, new_s_all = as_SE3_s_cuda(new_poses_sim3)  #new pfoses and scales
        new_poses_so3 = as_SO3_cuda(new_poses_sim3)
        
        last_poses_sim3 = lietorch.Sim3(torch.stack([keyframes[i].last_T_WC.data for i in range(len_keyframes)]))
        last_poses_se3, last_s_all = as_SE3_s_cuda(last_poses_sim3)
        last_poses_so3 = as_SO3_cuda(last_poses_sim3)
        
        

        with torch.no_grad():
            #tstamps = packet['tstamp']
            indices = self.gaussians.unique_kfIDs #no last keyframe
            #updates = packet['pose_updates'].cuda()[indices]
            #updates_scale = packet['scale_updates'].cuda()[indices]

            gaussian_scales = self.gaussians.get_scaling
            xyz = self.gaussians.get_xyz
            rot = SO3(self.gaussians.get_rotation)

            gaussian_update_num = 0
            
            for i in range(len(self.kf_ids)):  #no last id, self.kf_ids is the range of images  
                if i == 0:
                    kf_id = self.kf_ids[i]
                    cur_frame = keyframes[i]
                    H = cur_frame.img_shape[0,0].item() ; W = cur_frame.img_shape[0,1].item()    
                    cur_depth = cur_frame.X_canon[:,2].reshape(H,W)*new_s_all[i,0].item()  #plus sclae
                    cur_depth = cur_depth.cpu().numpy()
                    cur_depth[self.viewpoints[kf_id].water_mask.cpu().squeeze()] = 0.
                    self.viewpoints[kf_id].depth = cur_depth
                    continue

                cur_new_T = new_poses_sim3[i] ; cur_last_T = last_poses_sim3[i]
                pose_delta = as_SE3(cur_new_T*cur_last_T.inv())

            
                delta_matrix = pose_delta.matrix()[0]
                translation_diff = torch.norm(delta_matrix[:3, 3])
                rot_euler_diff_deg = torch.arccos((torch.trace(delta_matrix[:3, :3]) - 1)*0.5) * 180 / 3.1415926
                kf_id = self.kf_ids[i] # has turned to images ranges
                #print("kf id is ", kf_id ," and rot update is ", rot_euler_diff_deg, "and trans update is ", translation_diff)

                if rot_euler_diff_deg > self.rot_thre or translation_diff > self.trans_thre: #

                    gaussian_update_num += 1
                    cur_new_s = new_s_all[i] ; cur_last_s = last_s_all[i]
                    
                    mask = (indices == kf_id)
                    cur_gaussian_scales = gaussian_scales[mask]
                    cur_xyz = xyz[mask]
                    cur_rot = rot[mask]

                    updated_gaussian_scales = cur_gaussian_scales*cur_new_s/cur_last_s
                    self.gaussians._scaling[mask] = self.gaussians.scaling_inverse_activation(updated_gaussian_scales)

                    cur_new_R = new_poses_so3[i:i+1]; cur_last_R = last_poses_so3[i:i+1]
                    updated_rot = cur_new_R*cur_last_R.inv()*cur_rot
                    self.gaussians._rotation[mask] = updated_rot.data

                    
                    updated_xyz = cur_new_T.act(cur_last_T.inv().act(cur_xyz))
                    self.gaussians._xyz[mask] = updated_xyz


                    shape_cur = self.gaussians.get_xyz[mask].shape[0]

                    self.gaussians.xyz_gradient_accum[mask] = self.gaussians.xyz_gradient_accum[mask]/self.xyz_gradient_attnratio
                    self.gaussians.denom[mask] = self.gaussians.denom[mask]/self.xyz_gradient_attnratio
                    self.gaussians.max_radii2D[mask] = torch.zeros((shape_cur), device="cuda")

                new_T_matrix = new_poses_se3[i].inv().matrix()
                self.viewpoints[kf_id].update_T(new_T_matrix)
                
                cur_frame = keyframes[i]
                H = cur_frame.img_shape[0,0].item() ; W = cur_frame.img_shape[0,1].item()    
                cur_depth = cur_frame.X_canon[:,2].reshape(H,W)*new_s_all[i,0].item()  #plus sclae
                cur_depth = cur_depth.cpu().numpy()
                cur_depth[self.viewpoints[kf_id].water_mask.cpu().squeeze()] = 0.
                self.viewpoints[kf_id].depth = cur_depth

            # merge gaussians in cur frame and last lc id
            self.merge_by_voxel(keyframes, indices, debug=False, config=self.config)

        
        if gaussian_update_num/len(self.kf_ids) > self.frame_up_ratio:
            if self.debug:
                self.gaussians.save_ply(f'{self.save_dir}/3dgs_beforemapping.ply')
            self.global_mapping()
            if self.debug:
                self.gaussians.save_ply(f'{self.save_dir}/3dgs_aftermapping.ply')

        #new_viewpoint = Camera.init_from_tracking(packet["images"][i]/255.0, packet["depths"][i], packet["normals"][i], w2c[i], idx, self.projection_matrix, self.K, tstamp)
        if len_keyframes <3 and not finalize:
            return
        
        if not finalize:
            new_frame = keyframes[len_keyframes-2]
        else:
            new_frame = keyframes[len_keyframes-1]

        H = new_frame.img_shape[0,0].item() ; W = new_frame.img_shape[0,1].item()
        new_viewpoint = Camera.init_from_tracking(new_frame, self.projection_matrix, new_frame.K, W, H, C_conf, new_frame.water_mask, None)

        new_frame_id = new_frame.frame_id

        with torch.no_grad():
            render_pkg = render(new_viewpoint, self.gaussians, antialiasing=self.antialiasing)
            cur_opacity = render_pkg["accumulation"]

            new_depthmap = new_viewpoint.depth.copy()
            invalid_mask = cur_opacity[0,...] > self.opacity_thre
            # print(f"newdepth shape = {new_depthmap.shape}")
            invalid_mask = invalid_mask.cpu().numpy()
            # print(f"invalid_mask shape={invalid_mask.shape}")
            new_depthmap[invalid_mask] = 0.0

            cur_gaussian_points = self.gaussians.get_xyz[render_pkg["visibility_filter"]]
            cur_gaussian_points = cur_gaussian_points.clone().detach()

        self.add_next_kf_render(new_frame_id, new_viewpoint, cur_gaussian_points, init = False, depth_map = new_depthmap, gsplat=self.gsplat_enable)
        
        #self.add_next_kf(new_frame_id, new_viewpoint, init = False, depth_map = new_viewpoint.depth )

        self.viewpoints[new_frame_id] = new_viewpoint
        self.current_window = [new_frame_id] + self.current_window[:-1] if len(self.current_window) > 10 else [new_frame_id] + self.current_window 
        self.kf_ids.append(new_frame_id)


        # new_frame_id2 = keyframes[len_keyframes-1].frame_id
        # self.last_keyid = new_frame_id2  #ahead of viewpoints one step
        # self.Nonkeyframe_viewpoints[new_frame_id2] = []

        cur_new_window = [new_frame_id]
        self.map_cur(cur_new_window, iters=self.new_mapping_iter)
        


        # self.map(self.current_window, iters=1, prune=True)

        if self.use_gui:
            gui_keyframes = [self.viewpoints[kf_idx] for kf_idx in self.current_window]
            current_window_dict = {}
            current_window_dict[self.current_window[0]] = self.current_window[1:]
            self.q_main2vis.put(
                gui_utils.GaussianPacket(
                    gaussians=clone_obj(self.gaussians),
                    current_frame=new_viewpoint,
                    keyframes=gui_keyframes,
                    kf_window=current_window_dict,
                    gtcolor=new_viewpoint.original_image,
                    gtdepth=new_viewpoint.depth))
            
        return
    
    def finalize(self):
        self.color_refinement(iteration_total=self.gaussians.max_steps)
        self.gaussians.save_ply(f'{self.save_dir}/3dgs_final.ply')

        poses_cw = []
        for view in self.viewpoints.values():
            T_w2c = np.eye(4)
            T_w2c[0:3, 0:3] = view.R.cpu().numpy()
            T_w2c[0:3, 3] = view.T.cpu().numpy()
            poses_cw.append(np.hstack(([view.tstamp], to_se3_vec(T_w2c))))
        poses_cw.sort(key=lambda x: x[0])
        return np.stack(poses_cw)
    
    # def add_nonkeyframe(self, idx, last_keyframe_id, delta_pose, cur_K, W, H):  #only add pose
    #     fx = cur_K[0,0].item();  fy = cur_K[1,1].item() ; cx = cur_K[0,2].item(); cy = cur_K[1,2].item()
    #     new_viewpoint = Nonkeyframe_Camera(idx, delta_pose, None, None, self.projection_matrix, fx, fy, cx, cy, focal2fov(fx, W), focal2fov(fy, H), H, W)
    #     self.Nonkeyframe_viewpoints[last_keyframe_id].append(new_viewpoint)

    @torch.no_grad()
    #def eval_rendering(self, gtimages, gtdepthdir, traj, kf_idx):
    def eval_rendering_all(self, datasets, save_dir, antialiasing=True):

        eval_ate(self.viewpoints, datasets, save_dir, iterations="after_opt")
        print("ATE eval done")
        eval_rendering(self.all_nonkf_frames, self.gaussians, datasets, save_dir, antialiasing=antialiasing,config=self.config )
        print("non keyframe eval done")
        eval_rendering_kf(self.viewpoints, self.gaussians, save_dir, antialiasing=antialiasing, iteration="after_opt",config=self.config)
        print("keyframe eval done")
    
    @torch.no_grad()
    def result_of_render_and_trj_save(self, datasets, save_dir):

        save_gaussians(self.gaussians, save_dir, "final_after_opt", final=True)
        print("3D GSply save path is :",save_dir)

        import mast3r_slam.evaluate as eval
        seq_name = datasets.dataset_path.stem
        eval.save_traj_wt(save_dir, f"{seq_name}.txt", self.viewpoints)
        print("keyframe traj save path is :",save_dir)

        save_render_keyframes(self.viewpoints, self.gaussians, save_dir, antialiasing=True, iteration="after_opt",config=self.config)
        print("render keyframe image save path is :",save_dir)

        save_render_noneframes(self.all_nonkf_frames, datasets, self.gaussians, save_dir, antialiasing=True)
        print("all render image save path is :",save_dir)

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None, gsplat=True):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map, gsplat=gsplat
        )

    def add_next_kf_render(self, frame_idx, viewpoint, ext_points, init=False, scale=2.0, depth_map=None, gsplat=True):  #opacity 1 H W
        self.gaussians.extend_from_pcd_seq_render(
            viewpoint, ext_points, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map, gsplat=gsplat
        )

    def reset(self):
        self.iteration_count = 0
        self.current_window = []
        self.initialized = False
        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(viewpoint, self.gaussians, antialiasing=self.antialiasing)
            (image, viewspace_point_tensor, visibility_filter, radii, depth, rgb_med, rgb_clr
             #n_touched
             ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["rgb_medium"],
                render_pkg["rgb_clear"],
                # render_pkg["n_touched"]
            )

            outputs = {"render": image, "depth": depth, "rgb_clear": rgb_clr}

            #loss_init = get_loss_mapping_all(self.config, outputs, viewpoint, depth_loss=True)
            #loss_init = get_loss_mapping_rgbd(self.config, image, depth, viewpoint, initialization=True)
            #loss_init = get_loss_mapping_rgb(self.config, image, viewpoint) #
            if self.use_depth_loss:
                loss_init = get_loss_mapping_rgbd_mask(self.config, image, rgb_med, depth, viewpoint, initialization=True) 
            else:
                loss_init = get_loss_mapping_rgb_mask(self.config, image, rgb_med, viewpoint) #
            #loss_init = get_loss_mapping_rgbd_mask(self.config, image, rgb_med, depth, viewpoint, initialization=True) 
            # loss_init = get_water_loss(self.config, image, gt_image = viewpoint.original_image.cuda())
            scaling = self.gaussians.get_scaling
            mean_scaling = scaling.mean(dim=1).view(-1, 1)
            #isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            isotropic_loss = torch.abs(scaling- mean_scaling)
            loss_init += 3 * isotropic_loss.mean()
            loss_init.backward()

           
            with torch.no_grad():

                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter 
                )

                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset:
                    self.gaussians.reset_opacity()

                
                self.gaussians.optimizer.step()

                self.gaussians.optimizer.zero_grad(set_to_none=True)

                self.gaussians.Medium_optimizer.step()
                self.gaussians.Medium_optimizer.zero_grad(set_to_none=True)


        self.viewpoints[0] = viewpoint ; self.current_window.append(0); self.kf_ids.append(0)
        # GUI data sync (if enabled)
        if self.use_gui:
            keyframes = [self.viewpoints[kf_idx] for kf_idx in self.current_window]
            #current_window_dict[self.current_window[0]] = self.current_window[1:]
            self.q_main2vis.put(
                gui_utils.GaussianPacket(
                    gaussians=clone_obj(self.gaussians),
                    current_frame=viewpoint,
                    keyframes=keyframes,
                    gtcolor=viewpoint.original_image,
                    gtdepth=viewpoint.depth))
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, iters, prune=False):

        if len(current_window) == 0:
            return
        
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx not in current_window_set:
                random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            self.iteration_count += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []

            viewpoints = viewpoint_stack + [random_viewpoint_stack[idx] for idx in torch.randperm(len(random_viewpoint_stack))[:2]]
            for viewpoint in viewpoints:
                render_pkg = render(viewpoint, self.gaussians, antialiasing=self.antialiasing)
                image, viewspace_point_tensor, visibility_filter, radii, depth, rgb_clr = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["rgb_clear"])
                
                outputs = {"render": image, "depth": depth, "rgb_clear": rgb_clr}

                #loss_mapping += get_loss_mapping_all(self.config, outputs, viewpoint, depth_loss=False) 
                loss_mapping += get_loss_mapping_rgb(self.config, image, viewpoint)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            co_visibility_filter = torch.zeros_like(visibility_filter_acm[0])
            for visibility_filter in visibility_filter_acm:
                co_visibility_filter = torch.logical_or(co_visibility_filter, visibility_filter)

            # scaling = self.gaussians.get_scaling
            # mean_scaling = scaling.mean(dim=1).view(-1, 1)
            # #isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            # isotropic_loss = torch.abs(scaling[co_visibility_filter] - mean_scaling[co_visibility_filter])
            # loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = self.iteration_count % self.gaussian_update_every == self.gaussian_update_offset
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )

                ## Opacity reset
                #self.gaussian_reset = 501
                # if (self.iteration_count % self.gaussian_reset) == 0 and (not update_gaussian):
                #     Log("Resetting the opacity of non-visible Gaussians")
                #     self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                # self.gaussians.update_learning_rate(self.iteration_count)

                self.gaussians.Medium_optimizer.step()
                self.gaussians.Medium_optimizer.zero_grad(set_to_none=True)

    def map_cur(self, current_window, iters):  #only map cur windows

        if len(current_window) == 0:
            return
        
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]

        for _ in range(iters):
            self.iteration_count += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []

            viewpoints = viewpoint_stack
            for viewpoint in viewpoints:
                render_pkg = render(viewpoint, self.gaussians, antialiasing=self.antialiasing)
                image, viewspace_point_tensor, visibility_filter, radii, depth, rgb_med, rgb_clr = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["rgb_medium"],
                    render_pkg["rgb_clear"])

                outputs = {"render": image, "depth": depth, "rgb_clear": rgb_clr}

                # loss_mapping += get_loss_mapping_all(self.config, outputs, viewpoint, depth_loss=False)
                #loss_mapping += get_loss_mapping_rgb(self.config, image, viewpoint)
                if self.use_depth_loss:
                    loss_mapping += get_loss_mapping_rgbd_mask(self.config, image, rgb_med, depth, viewpoint)
                else:
                    loss_mapping += get_loss_mapping_rgb_mask(self.config, image, rgb_med, viewpoint)
                #loss_mapping += get_loss_mapping_rgbd_mask(self.config, image, rgb_med, depth, viewpoint)
                # loss_mapping += get_water_loss(self.config, image, gt_image = viewpoint.original_image.cuda())

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            co_visibility_filter = torch.zeros_like(visibility_filter_acm[0])
            for visibility_filter in visibility_filter_acm:
                co_visibility_filter = torch.logical_or(co_visibility_filter, visibility_filter)

            scaling = self.gaussians.get_scaling
            mean_scaling = scaling.mean(dim=1).view(-1, 1)
            #isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            isotropic_loss = torch.abs(scaling[co_visibility_filter] - mean_scaling[co_visibility_filter])
            loss_mapping += 3 * isotropic_loss.mean()
            loss_mapping.backward()
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = self.iteration_count % self.gaussian_update_every == self.gaussian_update_offset
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )

                ## Opacity reset
                #self.gaussian_reset = 501
                # if (self.iteration_count % self.gaussian_reset) == 0 and (not update_gaussian):
                #     Log("Resetting the opacity of non-visible Gaussians")
                #     self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                # self.gaussians.update_learning_rate(self.iteration_count)
                self.gaussians.Medium_optimizer.step()
                self.gaussians.Medium_optimizer.zero_grad(set_to_none=True)

    def normal_mapping(self, current_window):

        if len(current_window) == 0:
            return
        
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window] #cur viewpoints setss
        random_viewpoint_stack = []

        current_window_set = set(current_window)

        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx not in current_window_set:
                random_viewpoint_stack.append(viewpoint)

        for _ in range(self.normal_mapping_iters):
            self.iteration_count += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            cur_viewpoint_stack = [viewpoint_stack[idx] for idx in torch.randperm(len(viewpoint_stack))[:self.normal_mapping_cur_frame]]
            past_viewpoint_stack = [random_viewpoint_stack[idx] for idx in torch.randperm(len(random_viewpoint_stack))[:self.normal_mapping_past_frame]]
            viewpoints = cur_viewpoint_stack + past_viewpoint_stack
           
            for viewpoint in viewpoints:
                render_pkg = render(viewpoint, self.gaussians, antialiasing=self.antialiasing)
                image, viewspace_point_tensor, visibility_filter, radii, depth, rgb_med, rgb_clr = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["rgb_medium"],
                    render_pkg["rgb_clear"])

                outputs = {"render": image, "depth": depth, "rgb_clear": rgb_clr}

                # loss_mapping += get_loss_mapping_all(self.config, outputs, viewpoint, depth_loss=False)
                if not self.use_depth_loss:
                    loss_mapping += get_loss_mapping_rgb_mask(self.config, image, rgb_med, viewpoint)
                else:
                    loss_mapping += get_loss_mapping_rgbd_mask(self.config, image, rgb_med, depth, viewpoint)
                #loss_mapping += get_loss_mapping_rgb(self.config, image, viewpoint)
                # loss_mapping += get_water_loss(self.config, image, gt_image=viewpoint.original_image.cuda())

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            co_visibility_filter = torch.zeros_like(visibility_filter_acm[0])
            for visibility_filter in visibility_filter_acm:
                co_visibility_filter = torch.logical_or(co_visibility_filter, visibility_filter)

            scaling = self.gaussians.get_scaling
            mean_scaling = scaling.mean(dim=1).view(-1, 1)
            #isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            isotropic_loss = torch.abs(scaling[co_visibility_filter] - mean_scaling[co_visibility_filter])
            loss_mapping += 3 * isotropic_loss.mean()
            loss_mapping.backward()
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = self.iteration_count % self.gaussian_update_every == self.gaussian_update_offset
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )

                ## Opacity reset
                #self.gaussian_reset = 501
                # if (self.iteration_count % self.gaussian_reset) == 0 and (not update_gaussian):
                #     Log("Resetting the opacity of non-visible Gaussians")
                #     self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                # self.gaussians.update_learning_rate(self.iteration_count)

                self.gaussians.Medium_optimizer.step()
                self.gaussians.Medium_optimizer.zero_grad(set_to_none=True)

    def merge_by_voxel(self,keyframes, indices, debug: bool = True, config=None):
        # merge gaussians in cur frame and last lc id
        with keyframes.lc_lock:
            lc_size = keyframes.lc_size.value
            if lc_size < config["training"]["min_lc_size"]:
                return
            lc_ids = keyframes.lc_ids[:lc_size].clone()
            keyframes.lc_size.value = 0
            keyframes.reset_lc_ids()
            
        merge_dict = defaultdict(list)
        for j in range(lc_size):
            lc_id = lc_ids[j, 0]
            value = lc_ids[j, 1]
            merge_dict[lc_id].append(value)    

        mask = torch.zeros_like(indices, dtype=torch.bool, device="cuda")
        all_ids = []# tensor list

        for lc_id, merge_ids in merge_dict.items():
            if int(lc_id) >= len(self.kf_ids):
                with keyframes.lc_lock:
                    for id in merge_ids:
                        keyframes.lc_ids[keyframes.lc_size.value, 0] = lc_id
                        keyframes.lc_ids[keyframes.lc_size.value, 1] = id
                        keyframes.lc_size.value += 1
                    continue
            else:
                all_ids.append(lc_id)
                all_ids.extend(merge_ids)

        if all_ids:
            for id in all_ids:
                id_frame = self.kf_ids[id.item()]
                id_tensor = torch.tensor(id_frame, device="cuda")
                indices = indices.to("cuda")
                mask |= (indices == id_tensor)
            # merge gaussians buy voxel
            if mask.any():
                voxel = Voxel(voxel_size=self.config["training"]["voxel_size"], device="cuda", mask=mask, gaussian=self.gaussians)
                print("begin merge by voxel scheme2")
                voxel.merge_by_voxel_scheme2(debug=debug)
            else:
                if debug:
                    print("[Voxel Merge] Skip empty mask.")
                return
            
    def merge_by_voxel_v2(self, keyframes, indices, debug: bool = True): # test
        with keyframes.lc_lock:
            lc_size = keyframes.lc_size.value
            if lc_size < 2:
                return
            lc_ids = keyframes.lc_ids[:lc_size].clone()
            keyframes.lc_size.value = 0
            keyframes.reset_lc_ids()

        merge_dict = defaultdict(list)
        for j in range(lc_size):
            lc_id = lc_ids[j, 0].item()
            value = lc_ids[j, 1].item()
            merge_dict[lc_id].append(value)

        all_ids = []
        for lc_id, merge_ids in merge_dict.items():
            if int(lc_id) >= len(self.kf_ids):
                with keyframes.lc_lock:
                    for _id in merge_ids:
                        keyframes.lc_ids[keyframes.lc_size.value, 0] = lc_id
                        keyframes.lc_ids[keyframes.lc_size.value, 1] = _id
                        keyframes.lc_size.value += 1
                continue
            else:
                all_ids.append(lc_id)
                all_ids.extend(merge_ids)

        if not all_ids:
            if debug:
                print("[Voxel Merge] Skip: empty all_ids.")
            return

        device = torch.device("cuda")
        indices = indices.to(device)
        mask = torch.zeros_like(indices, dtype=torch.bool, device=device)

        for _id in all_ids:
            id_frame = self.kf_ids[int(_id)]            
            id_tensor = torch.tensor(id_frame, device=device)
            mask |= (indices == id_tensor)

        if not mask.any():
            if debug:
                print("[Voxel Merge] Skip: empty mask.")
            return

        voxel = Voxel(
            voxel_size=self.config["training"]["voxel_size"],
            device=str(device),
            mask=mask,
            gaussian=self.gaussians,   
        )
        voxel.merge_by_voxel_scheme2(debug=debug)

    def global_mapping(self):
        Log("Perform GS global mapping")

        for _ in tqdm(range(self.glomap_outiters)):
            
            for start_id in range(0, len(self.kf_ids), self.glomap_window):
            
                for _ in range(self.glomap_initers):

                    self.iteration_count += 1
                    loss_mapping = 0
                    viewspace_point_tensor_acm = []
                    visibility_filter_acm = []
                    radii_acm = []

                    for view_id in range(start_id, min(start_id+self.glomap_window, len(self.kf_ids))):

                        viewpoint = self.viewpoints[self.kf_ids[view_id]]
                        render_pkg = render(viewpoint, self.gaussians, antialiasing=self.antialiasing)
                        image, viewspace_point_tensor, visibility_filter, radii, depth, rgb_med, rgb_clr = (
                            render_pkg["render"],
                            render_pkg["viewspace_points"],
                            render_pkg["visibility_filter"],
                            render_pkg["radii"],
                            render_pkg["depth"],
                            render_pkg["rgb_medium"],
                            render_pkg["rgb_clear"])

                        
                        outputs = {"render": image, "depth": depth, "rgb_clear": rgb_clr}

                        # loss_mapping += get_loss_mapping_all(self.config, outputs, viewpoint, depth_loss=False)
                        #loss_mapping += get_loss_mapping_rgb(self.config, image, viewpoint)
                        if not self.use_depth_loss:
                            loss_mapping += get_loss_mapping_rgb_mask(self.config, image, rgb_med, viewpoint)
                        else:
                            loss_mapping += get_loss_mapping_rgbd_mask(self.config, image, rgb_med, depth, viewpoint)
                        # loss_mapping += get_water_loss(self.config, image, gt_image=viewpoint.original_image.cuda())


                        viewspace_point_tensor_acm.append(viewspace_point_tensor)
                        visibility_filter_acm.append(visibility_filter)
                        radii_acm.append(radii)

                    co_visibility_filter = torch.zeros_like(visibility_filter_acm[0])
                    for visibility_filter in visibility_filter_acm:
                        co_visibility_filter = torch.logical_or(co_visibility_filter, visibility_filter)

                    scaling = self.gaussians.get_scaling
                    mean_scaling = scaling.mean(dim=1).view(-1, 1)
                    #isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
                    isotropic_loss = torch.abs(scaling[co_visibility_filter] - mean_scaling[co_visibility_filter])
                    loss_mapping += 3 * isotropic_loss.mean()
                    loss_mapping.backward()
                    ## Deinsifying / Pruning Gaussians
                    with torch.no_grad():
                        for idx in range(len(viewspace_point_tensor_acm)):
                            self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                                self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                                radii_acm[idx][visibility_filter_acm[idx]],
                            )
                            self.gaussians.add_densification_stats(
                                viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                            )

                        update_gaussian = self.iteration_count % self.global_gaussian_update_every == self.gaussian_update_offset
                        if update_gaussian:
                            self.gaussians.densify_and_prune(
                                self.opt_params.densify_grad_threshold,
                                self.gaussian_th,
                                self.gaussian_extent,
                                self.size_threshold,
                            )

                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none=True)
                        # self.gaussians.update_learning_rate(self.iteration_count)

                        self.gaussians.Medium_optimizer.step()
                        self.gaussians.Medium_optimizer.zero_grad(set_to_none=True)

    def color_refinement_origin(self, iteration_per_frame): # iteration_per_frame = color_refinement_iters: 100
        Log("Starting color refinement")

        opt_params = []
        for view in self.viewpoints.values():
            opt_params.append({
                    "params": [view.cam_rot_delta],
                    "lr": self.config["opt_params"]["pose_lr"],
                    "name": "rot_{}".format(view.uid)})
            opt_params.append({
                    "params": [view.cam_trans_delta],
                    "lr": self.config["opt_params"]["pose_lr"],
                    "name": "trans_{}".format(view.uid)})
            if self.config["training"]["compensate_exposure"]:
                opt_params.append({
                        "params": [view.exposure_a],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_a_{}".format(view.uid)})
                opt_params.append({
                        "params": [view.exposure_b],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_b_{}".format(view.uid)})
        self.keyframe_optimizers = torch.optim.Adam(opt_params)

        
        self.gaussians.xyz_gradient_accum = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
        self.gaussians.denom = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
        self.gaussians.max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")

        self.debug_showImage = False



        viewpoint_idx_stack = list(self.viewpoints.keys())  # all keyframe indices
        iteration_total = iteration_per_frame * len(viewpoint_idx_stack)  # total iterations across all keyframes
        
        for iteration in tqdm(range(iteration_total)):
            viewpoint_cam_idx = viewpoint_idx_stack[random.randint(0, len(viewpoint_idx_stack) - 1)]
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(viewpoint_cam, self.gaussians, antialiasing=self.antialiasing)
            (image, image_med, viewspace_point_tensor, visibility_filter, radii) = (
                    render_pkg["render"],
                    render_pkg["rgb_medium"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )
            
            image = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b # exposure compensation: affine (a * I + b)
            image_med = (torch.exp(viewpoint_cam.exposure_a)) * image_med + viewpoint_cam.exposure_b
            gt_image = viewpoint_cam.original_image.cuda()


            #loss = (1.0 - self.opt_params.lambda_dssim) * l1_loss(image, gt_image) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            water_mask = viewpoint_cam.water_mask.squeeze().unsqueeze(0)


            # if self.debug_showid and viewpoint_cam.uid == 18: 
            #     plt.figure(figsize=(18, 6))
            #     plt.subplot(1, 3, 1)
            #     # Ensure image is on CPU and properly formatted for imshow
            #     display_image = image.detach().cpu().numpy().transpose(1, 2, 0)
            #     # Clip values to [0, 1] if they are floats, or ensure correct dtype for uint8
            #     if display_image.dtype == np.float32 or display_image.dtype == np.float64:
            #         display_image = np.clip(display_image, 0, 1)
            #     elif display_image.dtype != np.uint8:
            #         display_image = (display_image * 255).astype(np.uint8)
            #     plt.imshow(display_image)
            #     plt.title(f"Rendered Image - ID: {viewpoint_cam.uid}")

            #     plt.subplot(1, 3, 2)
            #     ...



                # Debug visualization
            if self.debug_showImage and iteration % 100 ==0:
                plt.figure(figsize=(18, 6))
                plt.subplot(1, 3, 1)
                # Ensure image is on CPU and properly formatted for imshow
                display_image = image.detach().cpu().numpy().transpose(1, 2, 0)
                # Clip values to [0, 1] if they are floats, or ensure correct dtype for uint8
                if display_image.dtype == np.float32 or display_image.dtype == np.float64:
                    display_image = np.clip(display_image, 0, 1)
                elif display_image.dtype != np.uint8:
                    display_image = (display_image * 255).astype(np.uint8)
                plt.imshow(display_image)
                plt.title(f"Rendered Image - ID: {viewpoint_cam.uid}")

                plt.subplot(1, 3, 2)
                # Ensure gt_image is on CPU and properly formatted for imshow
                display_gt_image = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
                if display_gt_image.dtype == np.float32 or display_gt_image.dtype == np.float64:
                    display_gt_image = np.clip(display_gt_image, 0, 1)
                elif display_gt_image.dtype != np.uint8:
                    display_gt_image = (display_gt_image * 255).astype(np.uint8)
                plt.imshow(display_gt_image)
                plt.title(f"Original Image - ID: {viewpoint_cam.uid}")

                plt.subplot(1, 3, 3)
                # Ensure water_mask is on CPU and properly formatted for imshow
                display_water_mask = water_mask.detach().cpu().squeeze().numpy()
                plt.imshow(display_water_mask, cmap='gray')
                plt.title(f"Water Mask - ID: {viewpoint_cam.uid}")
                plt.show(block=True)

            if iteration < iteration_total*self.opt_params.densify_iter_ratio:
                loss = (1.0 - self.opt_params.lambda_dssim) * l1_loss_mask(image, image_med, water_mask, gt_image) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            else:
                loss = (1.0 - self.opt_params.lambda_dssim) * l1_loss(image, gt_image) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))

            #loss = (1.0 - self.opt_params.lambda_dssim) * l1_loss(image, gt_image) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            if iteration < iteration_total*self.opt_params.densify_iter_ratio:
                scaling = self.gaussians.get_scaling
                mean_scaling = scaling.mean(dim=1).view(-1, 1)
                # isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
                isotropic_loss = torch.abs(scaling[visibility_filter] - mean_scaling[visibility_filter])
                loss += 3 * isotropic_loss.mean()

            #self.isotropic_losses.append(loss.item())


            loss.backward()

                
            with torch.no_grad():



                if iteration < iteration_total*self.opt_params.densify_iter_ratio: #0.7*100*keyframe 70%优化结构
                   
                   
                   
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter],radii[visibility_filter],)
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    update_gaussian = iteration % self.refinement_gaussian_update_every == self.gaussian_update_offset

                    if update_gaussian and self.gaussians.get_xyz.shape[0]<self.max_gaussian_points:
                        self.gaussians.densify_and_prune(
                            self.opt_params.densify_grad_threshold,
                            self.gaussian_th_refine,
                            self.gaussian_extent_refine,
                            self.size_threshold_refine,
                        )

                    # if (iteration % self.gaussian_reset == 0 ) and (not update_gaussian):
                    #     self.gaussians.reset_opacity()
                # ---- optimizer ----
                # Optimizer step and zero gradients
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                lr = self.gaussians.update_learning_rate(iteration)

                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)

                self.gaussians.Medium_optimizer.step()
                self.gaussians.Medium_optimizer.zero_grad(set_to_none=True)

                if viewpoint_cam.uid != 0:
                    update_pose(viewpoint_cam)

            if self.use_gui and iteration % 500 == 0:
                self.q_main2vis.put(gui_utils.GaussianPacket(gaussians=clone_obj(self.gaussians)))

                #pbar.set_description(f"Global GS Refinement lr {lr:.3E} loss {loss.item():.3f}")

                iteration += 1
        Log("Map refinement done")

    def color_refinement(self, iteration_per_frame):  # iteration_per_frame = color_refinement_iters: 100
        Log("Starting color refinement")

        opt_params = []
        for view in self.viewpoints.values():
            opt_params.append({
                "params": [view.cam_rot_delta],
                "lr": self.config["opt_params"]["pose_lr"],
                "name": "rot_{}".format(view.uid)})
            opt_params.append({
                "params": [view.cam_trans_delta],
                "lr": self.config["opt_params"]["pose_lr"],
                "name": "trans_{}".format(view.uid)})
            if self.config["training"]["compensate_exposure"]:
                opt_params.append({
                    "params": [view.exposure_a],
                    "lr": self.config["opt_params"]["exposure_lr"],
                    "name": "exposure_a_{}".format(view.uid)})
                opt_params.append({
                    "params": [view.exposure_b],
                    "lr": self.config["opt_params"]["exposure_lr"],
                    "name": "exposure_b_{}".format(view.uid)})
        self.keyframe_optimizers = torch.optim.Adam(opt_params)

        self.gaussians.xyz_gradient_accum = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
        self.gaussians.denom = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
        self.gaussians.max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")

        viewpoint_idx_stack = list(self.viewpoints.keys())  # 所有关键帧列表
        iteration_total = iteration_per_frame * len(viewpoint_idx_stack)  # total iterations across all keyframes

        for iteration in (pbar := trange(1, iteration_total + 1)):

            viewpoint_cam_idx = viewpoint_idx_stack[random.randint(0, len(viewpoint_idx_stack) - 1)]
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(viewpoint_cam, self.gaussians, antialiasing=self.antialiasing,
                                config_sh_degree_interval=self.config_sh_degree_interval, iteration=iteration,
                                shrefine=True, clip_thresh=self.config["color_refine"]["clip_thresh"])
            

            (image, image_med, viewspace_point_tensor, visibility_filter, radii,depth) = (
                render_pkg["render"],
                render_pkg["rgb_medium"], 
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                # render_pkg["n_touched"]
            )

            image = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b # exposure compensation: affine (a * I + b)
            image_med = (torch.exp(viewpoint_cam.exposure_a)) * image_med + viewpoint_cam.exposure_b
            water_mask = viewpoint_cam.water_mask.squeeze().unsqueeze(0)
            
            
            gt_image = viewpoint_cam.original_image.cuda()

            # loss = (1.0 - self.opt_params.lambda_dssim) * l1_loss(image, gt_image) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            #loss = get_water_loss(rgb=image, gt_image=gt_image)
            loss = get_water_loss_mask(rgb=image, rgb_med=image_med, water_mask=water_mask, gt_image=gt_image)
            depth_loss = get_loss_mapping_rgbd(self.config, image, depth, viewpoint_cam, initialization=False)
            scaling = self.gaussians.get_scaling
            mean_scaling = scaling.mean(dim=1).view(-1, 1)
            # isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            isotropic_loss = torch.abs(scaling[visibility_filter] - mean_scaling[visibility_filter])
            loss += 10 * isotropic_loss.mean()+depth_loss

            self.isotropic_losses.append(loss.item())

            loss.backward()


            with torch.no_grad():
                # ---- gsplat densification ----
                # gsplat prune, split, and reset alpha densification
                if (iteration < iteration_total * self.opt_params.densify_iter_ratio and iteration % self.refine_every == 0):  # 0.7*100*keyframe 70%优化结构
                    # update gradient stats
                    # record max 2D radii per pixel
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter], )
                    # accumulate densification stats (xy gradients and observation counts)
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    # compute reset interval
                    reset_interval = self.reset_alpha_every * self.refine_every  # opacity reset interval
                    # do_densification = (
                    #         iteration < self.stop_split_at
                    #         and (iteration % reset_interval == len(viewpoint_idx_stack) + self.refine_every)# step%500>keyframe_num(10)+100
                    # )
                    do_densification = (
                            iteration < self.stop_split_at
                            and (iteration % reset_interval > len(viewpoint_idx_stack) + self.refine_every)# step%500>keyframe_num(10)+100
                            and self.gaussians.get_xyz.shape[0]<self.max_gaussian_points
                    )
                    if do_densification:# and self.gaussians.get_xyz.shape[0]<self.max_gaussian_points:
                        # perform densification
                        assert self.gaussians.xyz_gradient_accum is not None and torch.ones_like(
                            self.gaussians.xyz_gradient_accum) is not None and self.gaussians.max_radii2D is not None
                        # avg_grad_norm = (self.gaussians.xyz_gradient_accum / torch.ones_like(self.gaussians.denom)) * 0.5 * max(viewpoint_cam.image_height,viewpoint_cam.image_width)
                        avg_grad_norm = (self.gaussians.xyz_gradient_accum / (self.gaussians.denom + 1e-6)) * 0.5 * max(
                            viewpoint_cam.image_height, viewpoint_cam.image_width)
                        avg_grad_norm[avg_grad_norm.isnan()] = 0.0
                        # select high-gradient Gaussians
                        high_grads = (avg_grad_norm > self.densify_grad_thresh).squeeze() # high_grads.mean() ~0.0019
                        # select large Gaussians
                        splits = (self.gaussians.get_scaling.max(dim=-1).values > self.densify_size_thresh).squeeze() # [0.00096,0.0012,0.28]
                        if iteration < self.stop_screen_size_at: #
                            splits |= (self.gaussians.max_radii2D > self.split_screen_size).squeeze() # split if 2D projection exceeds threshold
                        splits &= high_grads # mask: both large and high gradient

                        nsamps = self.n_split_samples # split each Gaussian into n_split_samples

                        self.gaussians.split_gaussians(splits, nsamps) # split Gaussians

                        # recompute high_grads after split
                        grad_norm = (self.gaussians.xyz_gradient_accum /
                                     (self.gaussians.denom + 1e-6)).squeeze()
                        grad_norm = grad_norm * 0.5 * max(viewpoint_cam.image_height, viewpoint_cam.image_width)
                        high_grads = grad_norm >= self.densify_grad_thresh

                        # find small Gaussians (below densify_size_thresh) for cloning
                        dups = (self.gaussians.get_scaling.max(dim=-1).values <= self.densify_size_thresh).squeeze()
                        dups &= high_grads # only clone small Gaussians with high gradients
                        if dups.any():
                            self.gaussians.dup_gaussians(dups) # clone Gaussians

                        # cull split Gaussians if below stop_split_at
                        splits_mask = torch.cat(
                            (
                                splits, # which duplicated Gaussians to split
                                torch.zeros(
                                    nsamps * splits.sum() + dups.sum(),# include newly created Gaussians (split + cloned)
                                    device="cuda",
                                    dtype=torch.bool,
                                ),
                            )
                        )

                        # align splits_mask to current Gaussian count
                        N = self.gaussians.get_xyz.shape[0]  # current total Gaussians

                        if splits_mask.shape[0] < N: #理论上不应该进去,就不可能，这俩应该是一直一样的。assert (splits_mask.shape[0] ==N), f"N ({N}) > splits_mask.shape[0] ({splits_mask.shape[0]})"
                            pad = N - splits_mask.shape[0]
                            splits_mask = torch.cat([splits_mask,
                                                     splits_mask.new_zeros(pad)])
                        elif splits_mask.shape[0] > N: #理论上不应该进去,就不可能，这俩应该是一直一样的。
                            splits_mask = splits_mask[:N]

                        self.gaussians.cull_gaussians(splits_mask,
                                                      iteration,
                                                      self.stop_split_at,
                                                      self.cull_screen_size,
                                                      self.cull_scale_thresh,
                                                      self.cull_alpha_thresh,
                                                      self.cull_alpha_thresh_post,
                                                      self.refine_every,
                                                      self.reset_alpha_every,
                                                      self.stop_screen_size_at
                                                      )  # cull Gaussians

                    elif (iteration >= self.stop_split_at and self.continue_cull_post_densification) or (self.gaussians.get_xyz.shape[0] >= self.max_gaussian_points):
                        # cull if split stopped and continue_cull enabled, or max points exceeded
                        self.gaussians.cull_gaussians(None,
                                                      iteration,
                                                      self.stop_split_at,
                                                      self.cull_screen_size,
                                                      self.cull_scale_thresh,
                                                      self.cull_alpha_thresh,
                                                      self.cull_alpha_thresh_post,
                                                      self.refine_every,
                                                      self.reset_alpha_every,
                                                      self.stop_screen_size_at
                                                      )
                    # else:
                    #     # skip if culling not allowed

                    if (iteration < self.stop_split_at) and (iteration % self.refine_every == self.refine_every):
                        # reset opacity every refine_every steps
                        self.gaussians.reset_opacity(self.reset_alpha_thresh)
                        # reset_value = self.reset_alpha_thresh
                        # max_logit_value = torch.logit(torch.tensor(reset_value)).item()  # precompute on CPU
                        # self.gaussians._opacity.clamp_(max=max_logit_value)

                # Optimizer step and zero gradients
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                lr = self.gaussians.update_learning_rate(iteration)

                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)

                self.gaussians.Medium_optimizer.step()
                self.gaussians.Medium_optimizer.zero_grad(set_to_none=True)
                if viewpoint_cam.uid != 0:
                    update_pose(viewpoint_cam)

            if self.use_gui and iteration % 500 == 0:
                gui_keyframes = [self.viewpoints[kf_idx] for kf_idx in self.viewpoints.keys()]
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        keyframes=gui_keyframes))

            pbar.set_description(f"Global GS Refinement lr {lr:.3E} loss {loss.item():.3f}")

        Log("Map refinement done")

    def _optimize_iteration(self,iteration,pbar):
        viewpoint_cam_idx = self.viewpoint_idx_stack[random.randint(0, len(self.viewpoint_idx_stack) - 1)]

        viewpoint_cam = self.viewpoints[viewpoint_cam_idx]

        render_pkg = render(viewpoint_cam, self.gaussians, antialiasing=self.antialiasing,
                            config_sh_degree_interval=self.config_sh_degree_interval, iteration=iteration,
                            shrefine=True, clip_thresh=self.config["color_refine"]["clip_thresh"])
        (image, image_med, viewspace_point_tensor, visibility_filter, radii) = (
            render_pkg["render"],
            render_pkg["rgb_medium"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            # render_pkg["depth"],
            # render_pkg["n_touched"]
        )

        image = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b
        image_med = (torch.exp(viewpoint_cam.exposure_a)) * image_med + viewpoint_cam.exposure_b
        gt_image = viewpoint_cam.original_image.cuda()

        # loss = (1.0 - self.opt_params.lambda_dssim) * l1_loss(image, gt_image) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
        water_mask = viewpoint_cam.water_mask.squeeze().unsqueeze(0)
        if self.debug_showid and viewpoint_cam.uid == 18:
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            # Ensure image is on CPU and properly formatted for imshow
            display_image = image.detach().cpu().numpy().transpose(1, 2, 0)
            # Clip values to [0, 1] if they are floats, or ensure correct dtype for uint8
            if display_image.dtype == np.float32 or display_image.dtype == np.float64:
                display_image = np.clip(display_image, 0, 1)
            elif display_image.dtype != np.uint8:
                display_image = (display_image * 255).astype(np.uint8)
            plt.imshow(display_image)
            plt.title(f"Rendered Image - ID: {viewpoint_cam.uid}")

            plt.subplot(1, 3, 2)
            ...

            # Debug visualization
        if self.debug_showImage and iteration % 200 == 0:
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            # Ensure image is on CPU and properly formatted for imshow
            display_image = image.detach().cpu().numpy().transpose(1, 2, 0)
            # Clip values to [0, 1] if they are floats, or ensure correct dtype for uint8
            if display_image.dtype == np.float32 or display_image.dtype == np.float64:
                display_image = np.clip(display_image, 0, 1)
            elif display_image.dtype != np.uint8:
                display_image = (display_image * 255).astype(np.uint8)
            plt.imshow(display_image)
            plt.title(f"Rendered Image - ID: {viewpoint_cam.uid}")

            plt.subplot(1, 3, 2)
            # Ensure gt_image is on CPU and properly formatted for imshow
            display_gt_image = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
            if display_gt_image.dtype == np.float32 or display_gt_image.dtype == np.float64:
                display_gt_image = np.clip(display_gt_image, 0, 1)
            elif display_gt_image.dtype != np.uint8:
                display_gt_image = (display_gt_image * 255).astype(np.uint8)
            plt.imshow(display_gt_image)
            plt.title(f"Original Image - ID: {viewpoint_cam.uid}")

            plt.subplot(1, 3, 3)
            # Ensure water_mask is on CPU and properly formatted for imshow
            display_water_mask = water_mask.detach().cpu().squeeze().numpy()
            plt.imshow(display_water_mask, cmap='gray')
            plt.title(f"Water Mask - ID: {viewpoint_cam.uid}")
            plt.show(block=True)

        # if iteration < iteration_total * self.opt_params.densify_iter_ratio:
        #     loss = get_water_loss_mask(image, image_med, water_mask, gt_image)
        # else:
        #     loss = get_water_loss(rgb=image, gt_image=gt_image)
        #loss = get_water_loss(rgb=image, gt_image=gt_image)
        if iteration < self.iteration_total * 0.7:
            loss = get_water_loss_mask(rgb=image, rgb_med=image_med, water_mask=water_mask, gt_image=gt_image)
        else:
            loss = get_water_loss(rgb=image, gt_image=gt_image)
        
        if self.use_iso_loss and iteration < self.iteration_total * 0.7 :
            scaling = self.gaussians.get_scaling
            mean_scaling = scaling.mean(dim=1).view(-1, 1)
            # isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            isotropic_loss = torch.abs(scaling[visibility_filter] - mean_scaling[visibility_filter])
            loss += 3 * isotropic_loss.mean()

        # self.isotropic_losses.append(loss.item())

        loss.backward()

        with torch.no_grad():
            # ---- gsplat densification ----
            # gsplat prune, split, and reset alpha densification
            if iteration < self.iteration_total * 0.7 and iteration % self.refine_every == 0 :  # 0.7*100*keyframe 70%优化结构
                # update gradient stats
                # record max 2D radii per pixel
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter], )
                # accumulate densification stats (xy gradients and observation counts)
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # compute reset interval
                reset_interval = self.reset_alpha_every * self.refine_every  # reset_alpha_every每经过多少次细化步数，重置不透明度 refine_every每隔多少步进行一次高斯点的裁剪和密化

                do_densification = (
                        iteration < self.stop_split_at
                        and (iteration % reset_interval > len(self.viewpoint_idx_stack) + self.refine_every)
                        and self.gaussians.get_xyz.shape[0]<self.max_gaussian_points
                # step%500>keyframe_num(10)+100
                )
                if do_densification:
                    # perform densification
                    assert self.gaussians.xyz_gradient_accum is not None and torch.ones_like(
                        self.gaussians.xyz_gradient_accum) is not None and self.gaussians.max_radii2D is not None
                    # avg_grad_norm = (self.gaussians.xyz_gradient_accum / torch.ones_like(self.gaussians.denom)) * 0.5 * max(viewpoint_cam.image_height,viewpoint_cam.image_width)
                    avg_grad_norm = (self.gaussians.xyz_gradient_accum / (self.gaussians.denom + 1e-6)) * 0.5 * max(
                        viewpoint_cam.image_height, viewpoint_cam.image_width)
                    avg_grad_norm[avg_grad_norm.isnan()] = 0.0
                    # select high-gradient Gaussians
                    high_grads = (avg_grad_norm > self.densify_grad_thresh * 0.5).squeeze()  # high_grads.mean() ~0.0019
                    # select large Gaussians
                    splits = (self.gaussians.get_scaling.max(
                        dim=-1).values > self.densify_size_thresh * 0.5).squeeze()  # [0.00096,0.0012,0.28]
                    if iteration < self.stop_screen_size_at:  #
                        splits |= (self.gaussians.max_radii2D > self.split_screen_size).squeeze()  # split if 2D projection exceeds threshold
                    splits &= high_grads  # mask: both large and high gradient
                    if self.debug:
                        scales = self.gaussians.get_scaling.max(dim=-1).values
                        print("split gausssian numbers: ", splits.sum().item())  # for debug
                        print("gaussian scale max", scales.max().item(), "current scale thresh: ",
                              self.densify_size_thresh)
                        print("gaussian scale mean", scales.mean().item())
                        print("gaussian scale min", scales.min().item())
                        print("gaussian grad max", avg_grad_norm.max().item(), "current grad thresh: ",
                              self.densify_grad_thresh)
                        print("gaussian grad mean", avg_grad_norm.mean().item())
                        print("gaussian grad min", avg_grad_norm.min().item())
                    nsamps = self.n_split_samples  # split each Gaussian into n_split_samples
                    self.gaussians.split_gaussians(splits, nsamps) # split Gaussians
                    #self.gaussians.split_gaussians_along_axis(splits, samps=nsamps)  # split Gaussians along max scale axis
                    # recompute high_grads after split
                    grad_norm = (self.gaussians.xyz_gradient_accum /
                                 (self.gaussians.denom + 1e-6)).squeeze()
                    grad_norm = grad_norm * 0.5 * max(viewpoint_cam.image_height, viewpoint_cam.image_width)
                    high_grads = grad_norm >= self.densify_grad_thresh

                    # find small Gaussians (below densify_size_thresh) for cloning
                    dups = (self.gaussians.get_scaling.max(dim=-1).values <= self.densify_size_thresh).squeeze()
                    dups &= high_grads  # only clone small Gaussians with high gradients
                    if dups.any():  # too small 2-5 need change param
                        self.gaussians.dup_gaussians(dups)  # clone Gaussians

                    # cull split Gaussians if below stop_split_at
                    splits_mask = torch.cat(
                        (
                            # splits, # which duplicated Gaussians to split
                            torch.zeros(splits.shape[0], device="cuda", dtype=torch.bool, ),  # keep if not split
                            torch.zeros(
                                nsamps * splits.sum() + dups.sum(),
                                # include newly created Gaussians (split + cloned)
                                device="cuda",
                                dtype=torch.bool,
                            ),
                        )
                    )

                    # align splits_mask to current Gaussian count
                    N = self.gaussians.get_xyz.shape[0]  # current total Gaussians

                    if splits_mask.shape[
                        0] < N:  # should not happen: size mismatch
                        pad = N - splits_mask.shape[0]
                        splits_mask = torch.cat([splits_mask,
                                                 splits_mask.new_zeros(pad)])
                    elif splits_mask.shape[0] > N:  # should not happen: size mismatch
                        splits_mask = splits_mask[:N]

                    self.gaussians.cull_gaussians(splits_mask,
                                                  iteration,
                                                  self.stop_split_at,
                                                  self.cull_screen_size,
                                                  self.cull_scale_thresh,
                                                  self.cull_alpha_thresh,
                                                  self.cull_alpha_thresh_post,
                                                  self.refine_every,
                                                  self.reset_alpha_every,
                                                  self.stop_screen_size_at
                                                  )  # cull Gaussians

                elif iteration >= self.stop_split_at and self.continue_cull_post_densification or (self.gaussians.get_xyz.shape[0] >= self.max_gaussian_points):
                    # continue culling after stop_split_at if configured
                    self.gaussians.cull_gaussians(None,
                                                  iteration,
                                                  self.stop_split_at,
                                                  self.cull_screen_size,
                                                  self.cull_scale_thresh,
                                                  self.cull_alpha_thresh,
                                                  self.cull_alpha_thresh_post,
                                                  self.refine_every,
                                                  self.reset_alpha_every,
                                                  self.stop_screen_size_at
                                                  )
                # else:
                #     # skip if culling not allowed

                if (iteration < self.stop_split_at) and (iteration % self.refine_every == self.refine_every):
                    # reset opacity every refine_every steps
                    self.gaussians.reset_opacity(self.reset_alpha_thresh)
                    # reset_value = self.reset_alpha_thresh
                    # max_logit_value = torch.logit(torch.tensor(reset_value)).item()  # precompute on CPU
                    # self.gaussians._opacity.clamp_(max=max_logit_value)

            # Optimizer step and zero gradients
            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            lr = self.gaussians.update_learning_rate(iteration)

            self.keyframe_optimizers.step()
            self.keyframe_optimizers.zero_grad(set_to_none=True)

            self.gaussians.Medium_optimizer.step()
            self.gaussians.Medium_optimizer.zero_grad(set_to_none=True)
            if viewpoint_cam.uid != 0:
                update_pose(viewpoint_cam)

        if self.use_gui and iteration % 400 == 0:
            self.q_main2vis.put(gui_utils.GaussianPacket(gaussians=clone_obj(self.gaussians)))

        iteration += 1
        pbar.set_postfix(lr=f"{lr:.3E}", loss=f"{loss.item():.3f}")

    def kf_pose_refinement(self, iteration_per_frame):
        Log("Starting kf pose refinement")

        for kf_id in tqdm(self.viewpoints.keys()):
            cur_viewpoint = self.viewpoints[kf_id]
            opt_params = []
            
            opt_params.append({
                    "params": [cur_viewpoint.cam_rot_delta],
                    "lr": self.config["opt_params"]["pose_lr"],
                    "name": "rot_{}".format(cur_viewpoint.uid)})
            opt_params.append({
                    "params": [cur_viewpoint.cam_trans_delta],
                    "lr": self.config["opt_params"]["pose_lr"],
                    "name": "trans_{}".format(cur_viewpoint.uid)})
            if self.config["training"]["compensate_exposure"]:
                opt_params.append({
                        "params": [cur_viewpoint.exposure_a],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_a_{}".format(cur_viewpoint.uid)})
                opt_params.append({
                        "params": [cur_viewpoint.exposure_b],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_b_{}".format(cur_viewpoint.uid)})
                
            cur_keyframe_optimizers = torch.optim.Adam(opt_params)

            gt_image = cur_viewpoint.original_image.cuda()

            if self.debug_showImage:
                with torch.no_grad():
                    render_pkg = render(cur_viewpoint, self.gaussians, antialiasing=self.antialiasing)
                    image = render_pkg["render"]

                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    # Ensure image is on CPU and properly formatted for imshow
                    display_image = image.detach().cpu().numpy().transpose(1, 2, 0)
                    # Clip values to [0, 1] if they are floats, or ensure correct dtype for uint8
                    if display_image.dtype == np.float32 or display_image.dtype == np.float64:
                        display_image = np.clip(display_image, 0, 1)
                    elif display_image.dtype != np.uint8:
                        display_image = (display_image * 255).astype(np.uint8)
                    plt.imshow(display_image)
                    plt.title(f"Rendered Image before pose refine - ID: {cur_viewpoint.uid}")

                    plt.subplot(1, 2, 2)
                    # Ensure gt_image is on CPU and properly formatted for imshow
                    display_gt_image = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
                    if display_gt_image.dtype == np.float32 or display_gt_image.dtype == np.float64:
                        display_gt_image = np.clip(display_gt_image, 0, 1)
                    elif display_gt_image.dtype != np.uint8:
                        display_gt_image = (display_gt_image * 255).astype(np.uint8)
                    plt.imshow(display_gt_image)
                    plt.title(f"Original Image - ID: {cur_viewpoint.uid}")
                    plt.show(block=True)
                torch.cuda.empty_cache()

            for iteration in range(iteration_per_frame):
                
                render_pkg = render(cur_viewpoint, self.gaussians, antialiasing=self.antialiasing)

                #image = (torch.exp(cur_viewpoint.exposure_a)) * render_pkg["render"] + cur_viewpoint.exposure_b

                image = render_pkg["render"]
                loss = (1.0 - self.opt_params.lambda_dssim) * l1_loss(image, gt_image) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                #loss += get_loss_mapping_rgb(self.config, image, viewpoint_cam)
                loss.backward()

                with torch.no_grad():

                    cur_keyframe_optimizers.step()
                    cur_keyframe_optimizers.zero_grad(set_to_none=True)
                    converged = update_pose(cur_viewpoint)

                del image, render_pkg
                torch.cuda.empty_cache()

                if converged:
                    break

            # visualize after refinement
            if self.debug_showImage:
                with torch.no_grad():
                    render_pkg = render(cur_viewpoint, self.gaussians, antialiasing=self.antialiasing)

                    image = render_pkg["render"]

                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    # Ensure image is on CPU and properly formatted for imshow
                    display_image = image.detach().cpu().numpy().transpose(1, 2, 0)
                    # Clip values to [0, 1] if they are floats, or ensure correct dtype for uint8
                    if display_image.dtype == np.float32 or display_image.dtype == np.float64:
                        display_image = np.clip(display_image, 0, 1)
                    elif display_image.dtype != np.uint8:
                        display_image = (display_image * 255).astype(np.uint8)
                    plt.imshow(display_image)
                    plt.title(f"Rendered Image after pose refinement - ID: {cur_viewpoint.uid}")

                    plt.subplot(1, 2, 2)
                    # Ensure gt_image is on CPU and properly formatted for imshow
                    display_gt_image = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
                    if display_gt_image.dtype == np.float32 or display_gt_image.dtype == np.float64:
                        display_gt_image = np.clip(display_gt_image, 0, 1)
                    elif display_gt_image.dtype != np.uint8:
                        display_gt_image = (display_gt_image * 255).astype(np.uint8)
                    plt.imshow(display_gt_image)
                    plt.title(f"Original Image with {iteration} iters - ID: {cur_viewpoint.uid}")
                    plt.show(block=True)
            

        Log("kf pose refinement done")

    def kf_BA_refinement(self):

        Log("Starting kf BA refinement")

        opt_params = []
        
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in self.viewpoints.keys()] #cur viewpoints setss


        for viewpoint in viewpoint_stack:
            opt_params.append({
                    "params": [viewpoint.cam_rot_delta],
                    "lr": self.config["opt_params"]["pose_lr"],
                    "name": "rot_{}".format(viewpoint.uid)})
            opt_params.append({
                    "params": [viewpoint.cam_trans_delta],
                    "lr": self.config["opt_params"]["pose_lr"],
                    "name": "trans_{}".format(viewpoint.uid)})
            if self.config["training"]["compensate_exposure"]:
                opt_params.append({
                        "params": [viewpoint.exposure_a],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_a_{}".format(viewpoint.uid)})
                opt_params.append({
                        "params": [viewpoint.exposure_b],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_b_{}".format(viewpoint.uid)})
                
        cur_keypose_optimizer = torch.optim.Adam(opt_params)
        iter_total = (self.GSBA_iter_perframe*len(viewpoint_stack))//self.GSBA_window
            
        iteration = 0

        for _ in tqdm(range(iter_total)):
            iteration += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            cur_viewpoint_stack = [viewpoint_stack[idx] for idx in torch.randperm(len(viewpoint_stack))[:self.GSBA_window]]
            
            for viewpoint in cur_viewpoint_stack:
                render_pkg = render(viewpoint, self.gaussians, antialiasing=self.antialiasing)
                image, viewspace_point_tensor, visibility_filter, radii, depth, rgb_med, rgb_clr = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["rgb_medium"],
                    render_pkg["rgb_clear"])
                
                # loss_mapping += get_loss_mapping_all(self.config, outputs, viewpoint, depth_loss=False)
              
                # image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b # exposure compensation: affine (a * I + b)
                # rgb_med = (torch.exp(viewpoint.exposure_a)) * rgb_med + viewpoint.exposure_b
                
                if not self.use_depth_loss:
                    loss_mapping += get_loss_mapping_rgb_mask(self.config, image, rgb_med, viewpoint)
                else:
                    if viewpoint.depth is not None:
                        loss_mapping += get_loss_mapping_rgbd_mask(self.config, image, rgb_med, depth, viewpoint)
                    else:
                        loss_mapping += get_loss_mapping_rgb_mask(self.config, image, rgb_med, viewpoint)
   
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            co_visibility_filter = torch.zeros_like(visibility_filter_acm[0])
            for visibility_filter in visibility_filter_acm:
                co_visibility_filter = torch.logical_or(co_visibility_filter, visibility_filter)

            scaling = self.gaussians.get_scaling
            mean_scaling = scaling.mean(dim=1).view(-1, 1)
            #isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            isotropic_loss = torch.abs(scaling[co_visibility_filter] - mean_scaling[co_visibility_filter])
            loss_mapping += 3 * isotropic_loss.mean()
            loss_mapping.backward()
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = iteration % (self.gaussian_update_every//self.GSBA_window) == (self.gaussian_update_offset//self.GSBA_window)
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )

                ## Opacity reset
                #self.gaussian_reset = 501
                # if (self.iteration_count % self.gaussian_reset) == 0 and (not update_gaussian):
                #     Log("Resetting the opacity of non-visible Gaussians")
                #     self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)

                self.gaussians.Medium_optimizer.step()
                self.gaussians.Medium_optimizer.zero_grad(set_to_none=True)
                cur_keypose_optimizer.step()

                for viewpoint in cur_viewpoint_stack:
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
                
                if self.use_gui and iteration % 200 == 0:
                    gui_keyframes = [self.viewpoints[kf_idx] for kf_idx in self.viewpoints.keys()]
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=clone_obj(self.gaussians),
                            keyframes=gui_keyframes))

    def nonkf_refinement(self, dataset, iteration_per_frame, img_size=512):
        Log("Starting Nonkf refinement")

        for viewpoint in tqdm(self.all_nonkf_frames):

            opt_params = []
            
            opt_params.append({
                    "params": [viewpoint.cam_rot_delta],
                    "lr": 10*self.config["opt_params"]["pose_lr"],
                    "name": "rot_{}".format(viewpoint.uid)})
            opt_params.append({
                    "params": [viewpoint.cam_trans_delta],
                    "lr": 10*self.config["opt_params"]["pose_lr"],
                    "name": "trans_{}".format(viewpoint.uid)})
            if self.config["training"]["compensate_exposure"]:
                opt_params.append({
                        "params": [viewpoint.exposure_a],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_a_{}".format(viewpoint.uid)})
                opt_params.append({
                        "params": [viewpoint.exposure_b],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_b_{}".format(viewpoint.uid)})
                
            cur_keyframe_optimizers = torch.optim.Adam(opt_params)

            _, gt_image = dataset[viewpoint.uid]

                    
            gt_image = resize_img(gt_image, img_size)["unnormalized_img"]

            gt_image = torch.from_numpy(gt_image.copy())/255.0
            gt_image = gt_image.permute(2,0,1).cuda()


            # with torch.no_grad():

            #     render_pkg = render(viewpoint, self.gaussians, antialiasing=self.antialiasing)
            #     image =  render_pkg["render"]
            #     image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

            #     gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            #     pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            #         np.uint8
            #     )



            #     plt.figure(figsize=(12, 6))
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(pred)
            #     plt.title(f"Rendered Image - ID: {viewpoint.uid}")

            #     plt.subplot(1, 2, 2)
            #     # Ensure gt_image is on CPU and properly formatted for imshow
            #     plt.imshow(gt)
            #     plt.title(f"Original Image with - ID: {viewpoint.uid}")
            #     plt.show(block=True)

                
            for iteration in range(iteration_per_frame):
                
                render_pkg = render(viewpoint, self.gaussians, antialiasing=self.antialiasing)

                
                image =  render_pkg["render"]
                image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
                loss = (1.0 - self.opt_params.lambda_dssim) * l1_loss(image, gt_image) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                #loss += get_loss_mapping_rgb(self.config, image, viewpoint_cam)
                loss.backward()

                    
                with torch.no_grad():

                    cur_keyframe_optimizers.step()
                    cur_keyframe_optimizers.zero_grad(set_to_none=True)
                    converged = update_pose(viewpoint)

                # if converged:
                #     break

            # with torch.no_grad():

            #     render_pkg = render(viewpoint, self.gaussians, antialiasing=self.antialiasing)
            #     image =  render_pkg["render"]
            #     image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

            #     gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            #     pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            #         np.uint8
            #     )



            #     plt.figure(figsize=(12, 6))
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(pred)
            #     plt.title(f"Rendered Image - ID: {viewpoint.uid}")

            #     plt.subplot(1, 2, 2)
            #     # Ensure gt_image is on CPU and properly formatted for imshow
            #     plt.imshow(gt)
            #     plt.title(f"Original Image with - ID: {viewpoint.uid}")
            #     plt.show(block=True)

        Log("Nonkf refinement done")

    def initialize_map_for_refine_test(self, cur_frame_idx,viewpoint_cam):
        Log("Starting color refinement")

        opt_params = []
        for view in self.viewpoints.values():
            opt_params.append({
                "params": [view.cam_rot_delta],
                "lr": self.config["opt_params"]["pose_lr"],
                "name": "rot_{}".format(view.uid)})
            opt_params.append({
                "params": [view.cam_trans_delta],
                "lr": self.config["opt_params"]["pose_lr"],
                "name": "trans_{}".format(view.uid)})
            if self.config["training"]["compensate_exposure"]:
                opt_params.append({
                    "params": [view.exposure_a],
                    "lr": self.config["opt_params"]["exposure_lr"],
                    "name": "exposure_a_{}".format(view.uid)})
                opt_params.append({
                    "params": [view.exposure_b],
                    "lr": self.config["opt_params"]["exposure_lr"],
                    "name": "exposure_b_{}".format(view.uid)})
        # self.keyframe_optimizers = torch.optim.Adam(opt_params)

        self.gaussians.xyz_gradient_accum = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
        self.gaussians.denom = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
        self.gaussians.max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")




        for iteration in (pbar := trange(1, 1000 + 1)):
            render_pkg = render(viewpoint_cam, self.gaussians, antialiasing=self.antialiasing,
                                config_sh_degree_interval=self.config_sh_degree_interval, iteration=iteration,
                                shrefine=True, clip_thresh=self.config["color_refine"]["clip_thresh"])
            (image, viewspace_point_tensor, visibility_filter, radii) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                # render_pkg["depth"],
                # render_pkg["n_touched"]
            )

            gt_image = viewpoint_cam.original_image.cuda()

            loss = get_water_loss(rgb=image, gt_image=gt_image)

            scaling = self.gaussians.get_scaling
            mean_scaling = scaling.mean(dim=1).view(-1, 1)
            # isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            isotropic_loss = torch.abs(scaling[visibility_filter] - mean_scaling[visibility_filter])
            loss += 10 * isotropic_loss.mean()

            self.isotropic_losses.append(loss.item())

            loss.backward()

            with torch.no_grad():
                # ---- gsplat densification ----
                # gsplat prune, split, and reset alpha densification
                if (iteration < 1000 * 0.7 and iteration % 100 == 0):  # 0.7*100*keyframe 70%优化结构
                    # update gradient stats
                    # record max 2D radii per pixel
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter], )
                    # accumulate densification stats (xy gradients and observation counts)
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    # compute reset interval
                    reset_interval = 2 * 100  # opacity reset interval
                    # do_densification = (
                    #         iteration < self.stop_split_at
                    #         and (iteration % reset_interval == len(viewpoint_idx_stack) + self.refine_every)# step%500>keyframe_num(10)+100
                    # )
                    do_densification = (
                            iteration < self.stop_split_at
                            and (iteration % reset_interval > 1+ 100)
                        # step%500>keyframe_num(10)+100
                    )
                    if do_densification:
                        # perform densification
                        assert self.gaussians.xyz_gradient_accum is not None and torch.ones_like(
                            self.gaussians.xyz_gradient_accum) is not None and self.gaussians.max_radii2D is not None
                        # avg_grad_norm = (self.gaussians.xyz_gradient_accum / torch.ones_like(self.gaussians.denom)) * 0.5 * max(viewpoint_cam.image_height,viewpoint_cam.image_width)
                        avg_grad_norm = (self.gaussians.xyz_gradient_accum / (self.gaussians.denom + 1e-6)) * 0.5 * max(
                            viewpoint_cam.image_height, viewpoint_cam.image_width)
                        avg_grad_norm[avg_grad_norm.isnan()] = 0.0
                        # select high-gradient Gaussians
                        high_grads = (avg_grad_norm > self.densify_grad_thresh).squeeze()  # high_grads.mean() ~0.0019
                        # select large Gaussians
                        splits = (self.gaussians.get_scaling.max(
                            dim=-1).values > self.densify_size_thresh).squeeze()  # [0.00096,0.0012,0.28]
                        if iteration < self.stop_screen_size_at:  #
                            splits |= (
                                        self.gaussians.max_radii2D > self.split_screen_size).squeeze()  # split if 2D projection exceeds threshold
                        splits &= high_grads  # mask: both large and high gradient

                        nsamps = self.n_split_samples  # split each Gaussian into n_split_samples
                        # self.gaussians.split_gaussians(splits, nsamps) # split Gaussians
                        self.gaussians.split_gaussians_along_axis(splits, samps=nsamps)  # split Gaussians along max scale axis
                        # recompute high_grads after split
                        grad_norm = (self.gaussians.xyz_gradient_accum /
                                     (self.gaussians.denom + 1e-6)).squeeze()
                        grad_norm = grad_norm * 0.5 * max(viewpoint_cam.image_height, viewpoint_cam.image_width)
                        high_grads = grad_norm >= self.densify_grad_thresh

                        # find small Gaussians (below densify_size_thresh) for cloning
                        dups = (self.gaussians.get_scaling.max(dim=-1).values <= self.densify_size_thresh).squeeze()
                        dups &= high_grads  # only clone small Gaussians with high gradients
                        if dups.any():
                            self.gaussians.dup_gaussians(dups)  # clone Gaussians

                        # cull split Gaussians if below stop_split_at
                        splits_mask = torch.cat(
                            (
                                splits,  # which duplicated Gaussians to split
                                torch.zeros(
                                    nsamps * splits.sum() + dups.sum(),
                                    # include newly created Gaussians (split + cloned)
                                    device="cuda",
                                    dtype=torch.bool,
                                ),
                            )
                        )

                        # align splits_mask to current Gaussian count
                        N = self.gaussians.get_xyz.shape[0]  # current total Gaussians

                        if splits_mask.shape[
                            0] < N:  # should not happen: size mismatch
                            pad = N - splits_mask.shape[0]
                            splits_mask = torch.cat([splits_mask,
                                                     splits_mask.new_zeros(pad)])
                        elif splits_mask.shape[0] > N:  # should not happen: size mismatch
                            splits_mask = splits_mask[:N]

                        self.gaussians.cull_gaussians(splits_mask,
                                                      iteration,
                                                      self.stop_split_at,
                                                      self.cull_screen_size,
                                                      self.cull_scale_thresh,
                                                      self.cull_alpha_thresh,
                                                      self.cull_alpha_thresh_post,
                                                      self.refine_every,
                                                      self.reset_alpha_every,
                                                      self.stop_screen_size_at
                                                      )  # cull Gaussians

                    elif iteration >= self.stop_split_at and self.continue_cull_post_densification:
                        # continue culling after stop_split_at if configured
                        self.gaussians.cull_gaussians(None,
                                                      iteration,
                                                      self.stop_split_at,
                                                      self.cull_screen_size,
                                                      self.cull_scale_thresh,
                                                      self.cull_alpha_thresh,
                                                      self.cull_alpha_thresh_post,
                                                      self.refine_every,
                                                      self.reset_alpha_every,
                                                      self.stop_screen_size_at
                                                      )
                    # else:
                    #     # skip if culling not allowed

                    if (iteration < self.stop_split_at) and (iteration % self.refine_every == self.refine_every):
                        # reset opacity every refine_every steps
                        self.gaussians.reset_opacity()
                        # reset_value = self.reset_alpha_thresh
                        # max_logit_value = torch.logit(torch.tensor(reset_value)).item()  # precompute on CPU
                        # self.gaussians._opacity.clamp_(max=max_logit_value)

                # Optimizer step and zero gradients
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                lr = self.gaussians.update_learning_rate(iteration)

                self.gaussians.Medium_optimizer.step()
                self.gaussians.Medium_optimizer.zero_grad(set_to_none=True)
                update_pose(viewpoint_cam)

            if self.use_gui and iteration % 1 == 0:
                self.q_main2vis.put(gui_utils.GaussianPacket(gaussians=clone_obj(self.gaussians)))

            pbar.set_description(f"Global GS Refinement lr {lr:.3E} loss {loss.item():.3f}")

        Log("Map refinement done")

