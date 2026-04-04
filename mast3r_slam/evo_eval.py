import argparse
import pathlib
import numpy as np
import os
import json
from colmap_utils import read_cameras_binary, read_images_binary, qvec2rotmat
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.core import metrics, trajectory
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from errno import EEXIST
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_2d_trajectory_with_colormap(
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

    # vmin = np.min(error_values)
    # vmax = np.max(error_values)
    # norm = plt.Normalize(vmin=max(0.0, vmin), vmax=max(vmax, 1e-3))
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
    plt.show()
    plt.close()

def plot_3d_trajectory_with_colormap(
    traj_est_aligned,
    traj_ref,
    error_values,
    ape_stat,
    ape_stats,
    plot_dir,
    label="aligned", plot_mode="xyz"
):
    os.makedirs(plot_dir, exist_ok=True)

    positions = traj_est_aligned.positions_xyz  # [N, 3]
    points = positions.reshape(-1, 1, 3)  # [N, 1, 3]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)  # [N-1, 2, 3]

    norm = Normalize(vmin=ape_stats["min"], vmax=ape_stats["max"])
    lc = Line3DCollection(segments, cmap="jet", norm=norm)
    lc.set_array(np.array(error_values))
    lc.set_linewidth(2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"ATE RMSE: {ape_stat:.4f} m")

    if traj_ref is not None:
        ref_xyz = traj_ref.positions_xyz
        ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2], linestyle='--', color='gray', label="GT")

    ax.add_collection3d(lc)

    cbar = fig.colorbar(lc, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("ATE [m]")

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(os.path.join(plot_dir, f"evo_3dplot_{label}.png"), dpi=120)
    plt.show()

def evaluate_evo(poses_gt, poses_est, timestamp_ref, timestamp_gt, plot_dir, label, monocular=True):
    """
    :param poses_gt:
    :param poses_est:
    :param plot_dir:
    :param label:
    :param monocular:
    :return:ape_error
    """
    # traj_ref = PosePath3D(poses_se3=poses_gt)
    # traj_est = PosePath3D(poses_se3=poses_est)
    # we have align the traject, so we use the same timestamp
    #timestamps = np.arange(len(poses_gt))
    traj_ref = PoseTrajectory3D(poses_se3=poses_gt, timestamps=timestamp_ref)
    traj_est = PoseTrajectory3D(poses_se3=poses_est, timestamps=timestamp_gt)

    traj_est_aligned = trajectory.align_trajectory( # trajectory matching using evo's Umeyama algorithm
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    print("RMSE ATE \[m]", ape_stat)

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    ## Plot
    plot_2d_trajectory_with_colormap(
        traj_ref,
        traj_est_aligned,
        ape_metric.error,
        ape_stats,
        plot_dir,
        label,
        ape_stat
    )
    plot_3d_trajectory_with_colormap(
        traj_est_aligned=traj_est_aligned,
        traj_ref=traj_ref,
        error_values=ape_metric.error,
        ape_stat=ape_stat,
        ape_stats=ape_stats,
        plot_dir=plot_dir,
        label=label
    )
    return ape_stat

def eval_ate(frames, datasets, save_dir, iterations, final=False, monocular=True):

    sorted_frames_index = sorted(frames)
    trj_data = dict()
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    for index, kf_index in enumerate(sorted_frames_index): # The number of trajectory points evaluated should be smaller than GT (from colmap)
        pose_est = frames[kf_index].T_WC
        pose_gt = datasets.get_pose(kf_index)

        trj_id.append(index)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est.cpu().numpy())
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    try:
        os.makedirs(plot_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST:
            pass
        else:
            raise

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
    return ate

def read_txt(traj_data_path):
    """

    :param traj_data_path: Quaternions need change to T_WC
    :return:pose and id list
    """
    pose_list = []
    image_id_list = []
    pose_dict = {}
    with open(traj_data_path,"r") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if not parts:
            continue
        image_id = int(float(parts[0]))
        t = np.array([float(x) for x in parts[1:4]])
        q = np.array([float(x) for x in parts[4:]])
        R = qvec2rotmat(q)
        T_WC = np.eye(4)
        T_WC[:3, :3] = R
        T_WC[:3, 3] = t
        pose_list.append(T_WC)
        image_id_list.append(image_id)
        pose_dict[image_id] = T_WC
    return pose_list, image_id_list, pose_dict

def read_colmap(colmap_path,sparse_dir:"sparse/0",img_dir:"images"):
    images = read_images_binary(colmap_path /sparse_dir/ "images.bin")
    pose_list = dict()
    pose_dict = {}
    for _, img in images.items():
        q = img.qvec  # [qw, qx, qy, qz]
        t = img.tvec  # [tx, ty, tz]
        R = qvec2rotmat(q)  # 3x3
        T_CW = np.eye(4)
        T_CW[:3, :3] = R
        T_CW[:3, 3] = t
        T_WC = np.linalg.inv(T_CW)
        pose_list[img.name] = T_WC
    all_items = sorted(os.listdir(colmap_path/img_dir))
    colmap_poses = [pose_list.get(p, np.eye(4)) for p in all_items]
    i = 0
    for elem in colmap_poses:
        pose_dict[i] = elem
        i += 1
    return colmap_poses,pose_dict


def read_colmap2(colmap_path,sparse_dir:"sparse/0",img_dir:"images"):
    images = read_images_binary(colmap_path /sparse_dir/ "images.bin")
    pose_list = []
    for _, img in images.items():
        q = img.qvec  # [qw, qx, qy, qz]
        t = img.tvec  # [tx, ty, tz]
        R = qvec2rotmat(q)  # 3x3
        T_CW = np.eye(4)
        T_CW[:3, :3] = R
        T_CW[:3, 3] = t
        T_WC = np.linalg.inv(T_CW)
        pose_list.append(T_WC)
    return pose_list

def align_timestamps(gt_poses:dict,ref_poses:dict):
    """
    :param gt_poses: number of pose is larger than ref_pose
    :param ref_poses:
    :return: align_gt_pose and align_ref_pose
    """
    align_gt_pose = [] # np list
    align_ref_pose = [] # np list
    for key in ref_poses.keys():
        align_ref_pose.append(ref_poses[key])
        align_gt_pose.append(gt_poses[key])
    return align_gt_pose,align_ref_pose

def align_namestamps(gt_poses:dict,ref_poses:dict):
    """
    :param gt_poses: number of pose is larger than ref_pose
    :param ref_poses:
    :return: align_gt_pose and align_ref_pose
    """
    if len(gt_poses) != len(ref_poses):
        print("Warning: The number of poses in gt_poses and ref_poses are not equal, please check!")
    align_gt_pose = [] # np list
    align_ref_pose = [] # np list

    for val in gt_path.values():
        align_gt_pose.append(val)
    for val in ref_poses.values():
        align_ref_pose.append(val)
    return align_gt_pose,align_ref_pose


if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument("--colmap_pose_dir", default=None)
    params.add_argument("--ref_pose_dir", default=None)
    params.add_argument("--save_dir", default=None)
    params.add_argument("--data_name", default="test")
    params.add_argument("--image_subdir",default="images")
    params.add_argument("--sparse_dir",default="sparse/0")
    args = params.parse_args()

    colmap_dir = args.colmap_pose_dir
    ref_dir = args.ref_pose_dir
    sparse_dir = args.sparse_dir
    image_subdir = args.image_subdir
    plot_dir = args.save_dir
    label_evo = args.data_name

    colmap_dir = "/home/robolab/Watersplatting-SLAM/ours_underwater/17_blue_rov/undistorted5"
    ref_dir = "/home/robolab/HI-SLAM2/outputs/undistorted5/traj_kf.txt"
    sparse_dir = "sparse/0"
    image_subdir = "images"
    plot_dir = "/home/robolab/HI-SLAM2/outputs/undistorted5/"
    label_evo = "undistorted5"


    _, gt_dicts = read_colmap(pathlib.Path(colmap_dir),sparse_dir,image_subdir)
    _, ref_image_idlists, ref_pose_dict = read_txt(ref_dir)
    gt_poses, ref_poses = align_timestamps(gt_dicts, ref_pose_dict)



    
    ate = evaluate_evo(
        poses_gt=gt_poses,
        poses_est=ref_poses,
        timestamp_gt=np.arange(len(gt_poses)),
        timestamp_ref=ref_image_idlists,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=True,
    )

