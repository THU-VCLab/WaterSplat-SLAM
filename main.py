import argparse
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp
from mast3r_slam.wt_gs_backend import WTGSBackEnd
from gaussian.utils.graphics_utils import focal2fov
from mast3r_slam.lietorch_utils import as_SE3_s
from gaussian.utils.camera_utils import Nonkeyframe_Camera
from mast3r_slam.segmodel import WaterSegmenter


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K):

    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)  #loop closure ids

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)

        with keyframes.lc_lock:
            for lc_id in lc_inds: # merge loop closure ids
                if abs(idx - lc_id) > config["reloc"]["min_lc_dist"]:# if the current id is too far away with the loop closure id, we do not merge 
                    keyframes.lc_ids[keyframes.lc_size.value, 0] = idx
                    keyframes.lc_ids[keyframes.lc_size.value, 1] = lc_id
                    keyframes.lc_size.value += 1

        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)


def run_wt_gs_backend(cfg, gs_backend: WTGSBackEnd, states, keyframes, dataset, origin_K):
    set_global_config(cfg)

    C_conf = config["training"]["C_conf"]
    color_refinement_iters = config["color_refine"]["color_refinement_iters"]
    nonkf_refinement_iters = config["training"]["nonkf_refinement_iters"]
    save_dir = config["output"]["base_dir"]

    mode = states.get_mode()
    resize_H = None; resize_W = None; cur_K = None

    gs_initialized = False
    gs_backend.datasets = dataset
    gs_backend.origin_K = origin_K

    origin_H, origin_W = dataset[0][1].shape[0:2]
    gs_backend.origin_width = origin_W; gs_backend.origin_height = origin_H

    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        idx = -1
        nonkeyframe_idx = -1; delta_pose = None

        with states.lock:
            if len(states.gs_optimizer_tasks) > 0:
                idx = states.gs_optimizer_tasks[0]

        while len(states.gs_nonkeyframe_tasks) > 0:
            with states.lock:
                nonkeyframe_idx, delta_pose, last_keyid = states.gs_nonkeyframe_tasks.pop(0)
            fx = cur_K[0,0].item(); fy = cur_K[1,1].item(); cx = cur_K[0,2].item(); cy = cur_K[1,2].item()

            cur_nonkeyframe_viewpoint = Nonkeyframe_Camera(nonkeyframe_idx, delta_pose.cuda(), None, \
                gs_backend.projection_matrix, fx, fy, cx, cy, focal2fov(fx, resize_W), focal2fov(fy, resize_H), resize_H, resize_W)

            gs_backend.add_new_nonkf(cur_nonkeyframe_viewpoint, last_keyid)

        if idx == -1:
            if gs_initialized:
                gs_backend.normal_mapping(gs_backend.current_window)
            time.sleep(0.01)
            continue

        if idx == 0:
            with states.lock:
                idx = states.gs_optimizer_tasks.pop(0)
            cur_frame = keyframes[0]
            resize_H = cur_frame.img_shape[0,0].item(); resize_W = cur_frame.img_shape[0,1].item(); cur_K = cur_frame.K
            gs_backend.init_projection_matrix(cur_K, resize_W, resize_H)
            continue

        gs_backend.process_track_data(keyframes, C_conf, gs_initialized)
        gs_initialized = True

        with states.lock:
            if len(states.gs_optimizer_tasks) > 0:
                idx = states.gs_optimizer_tasks.pop(0)

    while len(states.gs_nonkeyframe_tasks) > 0:
        with states.lock:
            nonkeyframe_idx, delta_pose, last_keyid = states.gs_nonkeyframe_tasks.pop(0)
        fx = cur_K[0,0].item(); fy = cur_K[1,1].item(); cx = cur_K[0,2].item(); cy = cur_K[1,2].item()

        cur_nonkeyframe_viewpoint = Nonkeyframe_Camera(nonkeyframe_idx, delta_pose.cuda(), None, \
            gs_backend.projection_matrix, fx, fy, cx, cy, focal2fov(fx, resize_W), focal2fov(fy, resize_H), resize_H, resize_W)

        gs_backend.add_new_nonkf(cur_nonkeyframe_viewpoint, last_keyid)

    gaussian_size, cur_gsmemory = gs_backend.gaussians.compute_gaussian_MB()
    print("Final GS memory: ", cur_gsmemory, "and gaussian size is ", gaussian_size)

    gs_backend.process_track_data(keyframes, C_conf, gs_initialized, finalize=True)

    gs_backend.get_all_viewpoints()
    gs_backend.insert_kf_all()

    if gs_backend.use_kf_BA:
        gs_backend.kf_BA_refinement()

    gs_backend.color_refinement_origin(color_refinement_iters)
    gaussian_size, cur_gsmemory = gs_backend.gaussians.compute_gaussian_MB()
    print("After refinement, Final GS memory: ", cur_gsmemory, "and gaussian size is ", gaussian_size)
    gs_backend.update_nonkf_pose()

    gs_backend.get_all_nonkf()

    gs_backend.nonkf_refinement(dataset, nonkf_refinement_iters)
    gs_backend.eval_rendering_all(dataset, save_dir, antialiasing=gs_backend.antialiasing)
    gs_backend.result_of_render_and_trj_save(dataset, save_dir)
    states.gs_refinement_done()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--calib", default="")
    parser.add_argument("--dataset_type", default=None,
                        help="Options: colmap, tum, euroc, eth3d, 7-scenes, realsense, webcam, mp4, rgbfiles")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-frames", action="store_true")
    return parser.parse_args()


def setup_dataset(args):
    """Load dataset, apply subsample, handle calibration. Returns (dataset, K, origin_K)."""
    dataset_type = args.dataset_type or config.get("dataset", {}).get("type", "auto")
    dataset = load_dataset(args.dataset, dataset_type=dataset_type)
    dataset.subsample(config["dataset"]["subsample"])

    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    K = None
    origin_K = None
    if config["use_calib"]:
        if not dataset.has_calib():
            print("[Warning] No calibration provided for this dataset!")
            sys.exit(0)
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            args.device, dtype=torch.float32
        )
        origin_K = dataset.camera_intrinsics.K

    return dataset, K, origin_K


def setup_processes(args, manager, model, states, keyframes, dataset, K, origin_K):
    """Spawn backend, gs_backend, and viz processes. Returns dict of process handles."""
    processes = {}

    # MAST3R Visualization
    if args.viz:
        main2viz = new_queue(manager, False)
        viz2main = new_queue(manager, False)
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()
        processes["viz"] = viz
    else:
        main2viz = new_queue(manager, True)
        viz2main = new_queue(manager, True)

    # SLAM backend
    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()
    processes["backend"] = backend

    # GS backend
    wt_gs_backend = WTGSBackEnd(config)
    segmodel = WaterSegmenter() if config["training"]["use_segmodel"] else None
    wt_gs_backend.segmodel = segmodel

    gs_proc = mp.Process(target=run_wt_gs_backend,
                         args=(config, wt_gs_backend, states, keyframes, dataset, origin_K))
    gs_proc.start()
    processes["gs_backend"] = gs_proc

    return processes, viz2main, segmodel


def run_tracking_loop(args, states, model, keyframes, dataset, tracker, viz2main, segmodel):
    """Main tracking loop. Returns (last_msg, frames)."""
    start_id = config["dataset"]["start_id"]
    final_id = config["dataset"]["final_id"]
    track_water_mask = config["training"]["track_water_mask"]
    text_prompt = config["training"]["text_prompt"]

    last_msg = WindowMsg()
    frames = []
    last_keyid = 0
    last_keyframe = None
    i = start_id
    fps_timer = None

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == min(len(dataset), final_id):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if args.save_frames:
            frames.append(img)
        add_new_kf = False

        T_WC = (
            lietorch.Sim3.Identity(1, device=args.device)
            if i == start_id
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size,
                             device=args.device, segmodel=segmodel, text_prompt=text_prompt)

        if mode == Mode.INIT:
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.gs_global_optimization(0)
            fps_timer = time.time()
            states.set_frame(frame)
            last_keyid = i
            T_WC_se, s = as_SE3_s(frame.T_WC)
            last_keyframe = T_WC_se[0].inv().matrix().cuda()
            states.set_mode(Mode.TRACKING)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame, track_water_mask)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

            if not add_new_kf:
                T_WC_se, s = as_SE3_s(frame.T_WC)
                cur_frame_pose = T_WC_se[0].inv().matrix().cuda()
                delta_pose = cur_frame_pose @ last_keyframe.inverse()
                states.gs_nonkeyframe_optimization(frame.frame_id, delta_pose.cpu(), last_keyid)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)

            last_keyid = i
            T_WC_se, s = as_SE3_s(frame.T_WC)
            last_keyframe = T_WC_se[0].inv().matrix().cuda()

            while True:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        states.gs_global_optimization(frame.frame_id)
                        break
                time.sleep(0.01)

            while True:
                with states.lock:
                    if len(states.gs_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)

        if i % 30 == 0 and fps_timer is not None:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    return last_msg, frames


def save_results(args, dataset, keyframes, last_msg, frames):
    """Save trajectory, reconstruction, keyframes, and debug frames."""
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        eval.save_reconstruction(
            save_dir,
            f"{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
        )
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )

    if args.save_frames and frames:
        base_dir = config["output"]["base_dir"]
        datetime_now = str(datetime.datetime.now()).replace(" ", "_")
        savedir = pathlib.Path(base_dir) / "frames" / datetime_now
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)


def main():
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    args = parse_args()
    load_config(args.config)
    # print(args.dataset)
    # print(config)

    # Setup dataset and calibration
    dataset, K, origin_K = setup_dataset(args)
    h, w = dataset.get_img_shape()[0]

    # Shared state
    manager = mp.Manager()
    keyframes = SharedKeyframes(manager, h, w, device=args.device)
    states = SharedStates(manager, h, w, device=args.device)
    if K is not None:
        keyframes.set_intrinsics(K)

    # Clean up previous results
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    # Load model
    model = load_mast3r(device=args.device)
    model.share_memory()

    # Spawn processes
    processes, viz2main, segmodel = setup_processes(
        args, manager, model, states, keyframes, dataset, K, origin_K
    )

    # Tracking
    tracker = FrameTracker(model, keyframes, args.device)
    last_msg, frames = run_tracking_loop(
        args, states, model, keyframes, dataset, tracker, viz2main, segmodel
    )

    # Wait for GS refinement to complete
    while states.gs_refinement_done_lists == 0:
        time.sleep(0.01)

    # Save results
    save_results(args, dataset, keyframes, last_msg, frames)

    print("done")

    # Wait for all child processes to finish before cleaning up shared CUDA tensors
    for name, p in processes.items():
        p.join()
        print(f"Process {name} joined")

    # Release shared CUDA tensors after all processes have exited
    del model, keyframes, tracker
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
