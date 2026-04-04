import copy
import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

#from gaussian.renderer import render
from water_gaussian.renderer import render
from gaussian.utils.image_utils import psnr
from gaussian.utils.loss_utils import ssim
from gaussian.utils.system_utils import mkdir_p
from gaussian.utils.logging_utils import Log
from mast3r_slam.mast3r_utils import resize_img
import subprocess
import mast3r_slam.colmap_utils as colmap
import pathlib
from mast3r_slam.config import config
import numpy as np
from matplotlib.collections import LineCollection



def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_trajectory_with_colormap(
        traj_ref,
        traj_est_aligned,
        ape_metric.error,
        ape_stats,
        plot_dir,
        label,
        ape_stat
    )
    return ape_stat

def plot_trajectory_with_colormap(
    traj_ref,
    traj_est_aligned,
    error_values,
    ape_stats,
    plot_dir,
    label,
    ape_stat,
    plot_mode=PlotMode.xy
):
    """
    Visualize the aligned trajectory and its error heatmap and save it as an image.

    :param traj_ref: ground truth trajectory (PosePath3D)
    :param traj_est_aligned: estimated trajectory after alignment (PosePath3D)
    :param error_values: list of error values for each frame
    :param ape_stats: error statistics (dict)
    :param plot_dir: save directory
    :param label: image save name suffix
    :param ape_stat: RMSE value (for title display)
    :param plot_mode: evo.plot.PlotMode (default xy plane)
    """
    os.makedirs(plot_dir, exist_ok=True)

    positions = traj_est_aligned.positions_xyz
    points = positions[:, :2].reshape(-1, 1, 2)  # [N, 1, 2]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)  # [N-1, 2, 2]

    norm = plt.Normalize(vmin=ape_stats["min"], vmax=ape_stats["max"])
    lc = LineCollection(segments, cmap="jet", norm=norm)
    lc.set_array(np.array(error_values))
    lc.set_linewidth(2)

    fig, ax = plt.subplots()
    ax.set_title(f"ATE RMSE: {ape_stat:.4f} m")

    plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")

    ax.add_collection(lc)

    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("ATE [m]")

    ax.legend()
    plt.savefig(os.path.join(plot_dir, f"evo_2dplot_{label}.png"), dpi=90)
    plt.close()

def eval_ate(frames, datasets, save_dir, iterations, final=False, monocular=True):
    sorted_frames_index = sorted(frames)
    trj_data = dict()
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    for index, kf_index in enumerate(sorted_frames_index): # The number of trajectory points evaluated should be smaller than GT (from colmap)
        pose_est = frames[kf_index].T_WC
        pose_gt = datasets.get_pose(kf_index)

        trj_id.append(kf_index)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est.cpu().numpy())
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    # return ate

def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    img_size=512,
    antialiasing=True,
    config = None
):
    interval = 1    
    end_idx = len(frames)
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")

    for idx in range(0, end_idx, interval):

        frame = frames[idx]
        _, gt_image = dataset[frame.uid]




        #gt_image = torch.from_numpy(gt_image)


        gt_image = resize_img(gt_image, img_size)["unnormalized_img"]
        gt_image = torch.from_numpy(gt_image.copy())/255.0



        
        gt_image = gt_image.permute(2,0,1).cuda()


        rendering = render(frame, gaussians, antialiasing=antialiasing)
        

        image = rendering["render"]
        unexpo_image = image
        unexpo_pred = (unexpo_image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )

        image = (torch.exp(frame.exposure_a)) * image + frame.exposure_b
        image = torch.clamp(image, 0.0, 1.0)
        
        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )

        if config["debug_config"]["debug_showImage"]:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(pred)
            plt.title(f"Rendered Image - ID: {frame.uid}")

            plt.subplot(1, 2, 2)
            # Ensure gt_image is on CPU and properly formatted for imshow
            plt.imshow(gt)
            plt.title(f"Original Image with - ID: {frame.uid}")
            plt.show(block=True)

            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            unexpo_pred = cv2.cvtColor(unexpo_pred, cv2.COLOR_BGR2RGB)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)


            image_outputpath = "outputs/images/gtimage_{}.png".format(frame.uid)
            unexpo_outputpath = "outputs/images/unexpo_{}.png".format(frame.uid)
            pred_outputpath = "outputs/images/pred_{}.png".format(frame.uid)
            cv2.imwrite(image_outputpath, gt)
            cv2.imwrite(unexpo_outputpath, unexpo_pred)
            cv2.imwrite(pred_outputpath, pred)
        

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())




        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )


    psnr_save_dir = os.path.join(save_dir, "psnr")
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output


def eval_rendering_kf(
    viewpoints,
    gaussians,
    save_dir,
    antialiasing=True,
    iteration="final",
    config=None
):
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
    for frame in viewpoints.values():
        gtimage = frame.original_image.cuda()

        rendering = render(frame, gaussians, antialiasing=antialiasing)

        image = rendering["render"]
        image = (torch.exp(frame.exposure_a)) * image + frame.exposure_b
        image = torch.clamp(image, 0.0, 1.0)


        gt = (gtimage.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )

        # gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        # pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        if config["debug_config"]["debug_showImage"]:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(pred)
            plt.title(f"Rendered Image - ID: {frame.uid}")

            plt.subplot(1, 2, 2)
            # Ensure gt_image is on CPU and properly formatted for imshow
            plt.imshow(gt)
            plt.title(f"Original Image with - ID: {frame.uid}")
            plt.show(block=True)


        # cv2.imshow("gtimage", gt)
        # cv2.imshow("pred", pred)
        # cv2.waitKey(0) ; cv2.destroyAllWindows()
        

        mask = gtimage > 0
        psnr_score = psnr((image[mask]).unsqueeze(0), (gtimage[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gtimage).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gtimage).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(f'kf mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}', tag="Eval")

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    os.makedirs(psnr_save_dir, exist_ok=True)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result_kf.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output

def save_render_keyframes(
    viewpoints,
    gaussians,
    save_dir,
    antialiasing=True,
    iteration="final",
    config=None
):
    for frame in viewpoints.values():
        rendering = render(frame, gaussians, antialiasing=antialiasing)
        image = rendering["render"]
        image = (torch.exp(frame.exposure_a)) * image + frame.exposure_b
        image = torch.clamp(image, 0.0, 1.0)
        image = (image.detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

        clean_image = rendering["rgb_clear"]
        clean_image = (torch.exp(frame.exposure_a)) * clean_image + frame.exposure_b
        clean_image = torch.clamp(clean_image, 0.0, 1.0)
        clean_image = (clean_image.detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

        clr_img = cv2.cvtColor(clean_image, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        path = os.path.join(save_dir, "rendering_keyframe", str(iteration), f"{frame.uid}.png")
        path_clr = os.path.join(save_dir, "rendering_keyframe_clr", str(iteration), f"{frame.uid}.png")
        mkdir_p(os.path.dirname(path))
        mkdir_p(os.path.dirname(path_clr))

        cv2.imwrite(path_clr, clr_img)
        cv2.imwrite(path,img)

def save_render_noneframes(
    frames,
    datasets,
    gaussians,
    save_dir,
    antialiasing=True,
):
    interval = 1    
    end_idx = len(frames)
    for idx in range(0, end_idx, interval):
        frame = frames[idx]
        rendering = render(frame, gaussians, antialiasing=antialiasing)
        image = rendering["render"]
        image = (torch.exp(frame.exposure_a)) * image + frame.exposure_b
        image = torch.clamp(image, 0.0, 1.0)
        image = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )

        clean_image = rendering["rgb_clear"]
        clean_image = (torch.exp(frame.exposure_a)) * clean_image + frame.exposure_b
        clean_image = torch.clamp(clean_image, 0.0, 1.0)
        clean_image = (clean_image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )

        clr_img = cv2.cvtColor(clean_image, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        path = os.path.join(save_dir, "rendering_nonekeyframe", str(frame.uid) + ".png")
        path_clr = os.path.join(save_dir, "rendering_nonekeyframe_clr", str(frame.uid) + ".png")
        mkdir_p(os.path.dirname(path))
        mkdir_p(os.path.dirname(path_clr))
        cv2.imwrite(path_clr, clr_img)
        cv2.imwrite(path, img)

        
def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


def imshow(image,title):
    import cv2
    import numpy as np
    image = image.detach().cpu().numpy()
    rgb_show = np.clip(image, 0, 1)  # 确保数值不越

    cv2.imshow(title, (rgb_show * 255).astype(np.uint8))
    cv2.waitKey(1)


if __name__ == "__main__":
    trj_gt_np = [
        np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ]),
        np.array([
            [0.96182513, -0.20738088, 0.17856538, -2.83185178],
            [0.23712227, 0.95728337, -0.1654738, 3.09259154],
            [-0.13662157, 0.20149869, 0.96991382, -1.69780678],
            [0., 0., 0., 1.]
        ]),
        np.array([
            [0.96182513, -0.20738088, 0.17856538, -2.83185178],
            [0.23712227, 0.95728337, -0.1654738, 3.09259154],
            [-0.13662157, 0.20149869, 0.96991382, -1.69780678],
            [0., 0., 0., 1.]
        ])
    ]

    trj_est_np = [
        np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ]),
        np.array([
            [0.96182513, -0.20738088, 0.17856538, -2.83185178],
            [0.23712227, 0.95728337, -0.1654738, 3.09259154],
            [-0.13662157, 0.20149869, 0.96991382, -1.69780678],
            [0., 0., 0., 1.]
        ])
    ]

    plot_dir = '/home/asus/slam/WaterSplatting-SLAM/outputs/Panama/plot'
    label_evo = 'after_opt'
    monocular = True

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    print("ATE RMSE:", ate)
