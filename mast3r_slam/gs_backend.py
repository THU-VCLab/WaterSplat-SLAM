import random
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import trange
from munch import munchify
from lietorch import SE3, SO3

from gaussian.utils.logging_utils import Log, clone_obj
from gaussian.renderer import render
from gaussian.utils.loss_utils import l1_loss, ssim
from gaussian.scene.gaussian_model import GaussianModel
from gaussian.utils.graphics_utils import getProjectionMatrix2
from gaussian.utils.pose_utils import update_pose
from gaussian.utils.slam_utils import to_se3_vec, get_loss_normal, get_loss_mapping_rgbd, get_loss_mapping_rgb
from gaussian.utils.camera_utils import Camera
from gaussian.utils.eval_utils import eval_rendering, eval_rendering_kf
from gui import gui_utils, slam_gui
from mast3r_slam.lietorch_utils import as_SE3_s_cuda, as_SO3_cuda, as_SE3
import lietorch
from tqdm import tqdm

class GSBackEnd(mp.Process):
    def __init__(self, config, save_dir, use_gui=False):
        super().__init__()
        self.config = config
        
        self.iteration_count = 0
        self.viewpoints = {}
        self.current_window = []
        self.initialized = False
        self.save_dir = save_dir
        self.use_gui = use_gui

        self.opt_params = munchify(config["opt_params"])
        self.config["training"]["monocular"] = False

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


        self.gaussians = GaussianModel(sh_degree=0, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        

        self.cameras_extent = 6.0
        self.projection_matrix = None

        self.kf_ids = []

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
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
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
        self.gaussian_update_offset = self.config["training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["training"]["gaussian_th"]
        self.gaussian_extent = self.cameras_extent * self.config["training"]["gaussian_extent"]
        self.gaussian_reset = self.config["training"]["gaussian_reset"]
        self.size_threshold = self.config["training"]["size_threshold"]
        self.window_size = self.config["training"]["window_size"]
        self.lambda_dnormal = self.config["training"]["lambda_dnormal"]

    def init_projection_matrix(self, frame, cur_K, W, H):
        
        self.projection_matrix = getProjectionMatrix2(znear=0.01, zfar=100.0, fx = cur_K[0,0].item(), \
            fy = cur_K[1,1].item() , cx = cur_K[0,2].item(), cy = cur_K[1,2].item(), W=W, H=H).transpose(0, 1).cuda()

    def process_track_data(self, keyframes, C_conf):
        #find old pose and cur poses
        #get new s and new poses
        len_keyframes = len(keyframes)
        new_poses_sim3 = lietorch.Sim3(torch.stack([keyframes[i].T_WC.data for i in range(len_keyframes)]))
        new_poses_se3, new_s_all = as_SE3_s_cuda(new_poses_sim3)  #new poses and scales
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
            
            for i in range(len(self.kf_ids)):  #no last id
                if i == 0:
                    kf_id = self.kf_ids[i]
                    cur_frame = keyframes[i]
                    H = cur_frame.img_shape[0,0].item() ; W = cur_frame.img_shape[0,1].item()    
                    cur_depth = cur_frame.X_canon[:,2].reshape(H,W)*new_s_all[i,0].item()  #plus sclae
                    cur_depth = cur_depth.cpu().numpy()
                    self.viewpoints[kf_id].depth = cur_depth
                    continue

                cur_new_T = new_poses_sim3[i] ; cur_last_T = last_poses_sim3[i]
                pose_delta = as_SE3(cur_new_T*cur_last_T.inv())

            
                delta_matrix = pose_delta.matrix()[0]
                translation_diff = torch.norm(delta_matrix[:3, 3])
                rot_euler_diff_deg = torch.arccos((torch.trace(delta_matrix[:3, :3]) - 1)*0.5) * 180 / 3.1415926
                kf_id = self.kf_ids[i]
                #print("kf id is ", kf_id ," and rot update is ", rot_euler_diff_deg, "and trans update is ", translation_diff)

                if rot_euler_diff_deg > self.rot_thre or translation_diff > self.trans_thre:

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
                self.viewpoints[kf_id].depth = cur_depth



        if gaussian_update_num/len(self.kf_ids) > self.frame_up_ratio:
            self.global_mapping()

        #new_viewpoint = Camera.init_from_tracking(packet["images"][i]/255.0, packet["depths"][i], packet["normals"][i], w2c[i], idx, self.projection_matrix, self.K, tstamp)
        if len_keyframes <3:
            return

        new_frame = keyframes[len_keyframes-2]
        H = new_frame.img_shape[0,0].item() ; W = new_frame.img_shape[0,1].item()    
        new_viewpoint = Camera.init_from_tracking(new_frame, self.projection_matrix, new_frame.K, W, H, C_conf, None)
        new_frame_id = new_frame.frame_id

        # render_pkg = render(new_viewpoint, self.gaussians, self.background)
        # cur_opacity = render_pkg["opacity"]

        # new_depthmap = new_viewpoint.depth.copy()
        # invalid_mask = cur_opacity[0] > self.opacity_thre
        # invalid_mask = invalid_mask.cpu().numpy()
        # new_depthmap[invalid_mask] = 0.0

        with torch.no_grad():
            render_pkg = render(new_viewpoint, self.gaussians, self.background)
            cur_opacity = render_pkg["opacity"]

            new_depthmap = new_viewpoint.depth.copy()
            invalid_mask = cur_opacity[0] > self.opacity_thre
            # print(f"newdepth shape = {new_depthmap.shape}")
            invalid_mask = invalid_mask.cpu().numpy()
            # print(f"invalid_mask shape={invalid_mask.shape}")
            new_depthmap[invalid_mask] = 0.0

            cur_gaussian_points = self.gaussians.get_xyz[render_pkg["visibility_filter"]]
            cur_gaussian_points = cur_gaussian_points.clone().detach()

        self.add_next_kf_render(new_frame_id, new_viewpoint, cur_gaussian_points, init = False, depth_map = new_depthmap )
        
        #self.add_next_kf(new_frame_id, new_viewpoint, init = False, depth_map = new_viewpoint.depth )

        self.viewpoints[new_frame_id] = new_viewpoint
        self.current_window = [new_frame_id] + self.current_window[:-1] if len(self.current_window) > 10 else [new_frame_id] + self.current_window 
        self.kf_ids.append(new_frame_id)

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

    @torch.no_grad()
    def eval_rendering(self, gtimages, gtdepthdir, traj, kf_idx):
        eval_rendering(gtimages, gtdepthdir, traj, self.gaussians,self.save_dir, self.background,
            self.projection_matrix, self.K, kf_idx, iteration="after_opt")
        eval_rendering_kf(self.viewpoints, self.gaussians, self.save_dir, self.background, iteration="after_opt")

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def add_next_kf_render(self, frame_idx, viewpoint, ext_points, init=False, scale=2.0, depth_map=None):  #opacity 1 H W
        self.gaussians.extend_from_pcd_seq_render(
            viewpoint, ext_points=ext_points, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
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
            render_pkg = render(viewpoint, self.gaussians, self.background)
            (image, viewspace_point_tensor, visibility_filter, radii, depth, n_touched) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["n_touched"]
            )
            #loss_init = get_loss_mapping_rgbd(self.config, image, depth, viewpoint, initialization=True)
            loss_init = get_loss_mapping_rgb(self.config, image, depth, viewpoint)
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

        self.viewpoints[0] = viewpoint ; self.current_window.append(0) ; self.kf_ids.append(0)
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
                render_pkg = render(viewpoint, self.gaussians, self.background)
                image, viewspace_point_tensor, visibility_filter, radii, depth, n_touched = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["n_touched"])

                loss_mapping += get_loss_mapping_rgb(self.config, image, depth, viewpoint)
                #loss_mapping += get_loss_mapping_rgbd(self.config, image, depth, viewpoint)
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
            loss_mapping += 10 * isotropic_loss.mean()
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
                render_pkg = render(viewpoint, self.gaussians, self.background)
                image, viewspace_point_tensor, visibility_filter, radii, depth, n_touched = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["n_touched"])

                loss_mapping += get_loss_mapping_rgb(self.config, image, depth, viewpoint)
                #loss_mapping += get_loss_mapping_rgbd(self.config, image, depth, viewpoint)
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
            loss_mapping += 10 * isotropic_loss.mean()
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
                render_pkg = render(viewpoint, self.gaussians, self.background)
                image, viewspace_point_tensor, visibility_filter, radii, depth, n_touched = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["n_touched"])

                loss_mapping += get_loss_mapping_rgb(self.config, image, depth, viewpoint)
                #loss_mapping += get_loss_mapping_rgbd(self.config, image, depth, viewpoint)
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
            loss_mapping += 10 * isotropic_loss.mean()
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

    def global_mapping(self):

        Log("Perform GS global mapping")
        self.gaussians.xyz_gradient_accum = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
        self.gaussians.denom = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
        self.gaussians.max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")

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
                        render_pkg = render(viewpoint, self.gaussians, self.background)
                        image, viewspace_point_tensor, visibility_filter, radii, depth, n_touched = (
                            render_pkg["render"],
                            render_pkg["viewspace_points"],
                            render_pkg["visibility_filter"],
                            render_pkg["radii"],
                            render_pkg["depth"],
                            render_pkg["n_touched"])

                        loss_mapping += get_loss_mapping_rgb(self.config, image, depth, viewpoint)
                        #loss_mapping += get_loss_mapping_rgbd(self.config, image, depth, viewpoint)
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
                    loss_mapping += 10 * isotropic_loss.mean()
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

                        ## Opacity reset
                        #self.gaussian_reset = 501
                        # if (self.iteration_count % self.gaussian_reset) == 0 and (not update_gaussian):
                        #     Log("Resetting the opacity of non-visible Gaussians")
                        #     self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)

                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none=True)
                        # self.gaussians.update_learning_rate(self.iteration_count)


    def color_refinement(self, iteration_total):
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

        for iteration in (pbar := trange(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(random.randint(0, len(viewpoint_idx_stack) - 1))
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(viewpoint_cam, self.gaussians, self.background)
            image, depth = render_pkg["render"], render_pkg["depth"]
            image = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b

            gt_image = viewpoint_cam.original_image.cuda()
            loss = (1.0 - self.opt_params.lambda_dssim) * l1_loss(image, gt_image) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss += get_loss_mapping_rgbd(self.config, image, depth, viewpoint_cam)
            if iteration < 7000:
                loss += self.lambda_dnormal * get_loss_normal(depth, viewpoint_cam)
            else:
                loss += self.lambda_dnormal * get_loss_normal(depth, viewpoint_cam) / 2
            loss.backward()
            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                lr = self.gaussians.update_learning_rate(iteration)

                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                update_pose(viewpoint_cam)

            if self.use_gui and iteration % 50 == 0:
                self.q_main2vis.put(gui_utils.GaussianPacket(gaussians=clone_obj(self.gaussians)))

            pbar.set_description(f"Global GS Refinement lr {lr:.3E} loss {loss.item():.3f}")

        Log("Map refinement done")
