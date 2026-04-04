import pathlib
import re
import cv2
from natsort import natsorted
import numpy as np
import torch
import yaml
import os

from mast3r_slam.mast3r_utils import resize_img
from mast3r_slam.config import config
from .colmap_utils import read_cameras_binary, read_images_binary, qvec2rotmat

HAS_TORCHCODEC = True
try:
    from torchcodec.decoders import VideoDecoder
except Exception:
    HAS_TORCHCODEC = False

HAS_REALSENSE = True
try:
    import pyrealsense2 as rs
except ImportError:
    HAS_REALSENSE = False


class MonocularDataset(torch.utils.data.Dataset):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
        self.rgb_files = []
        self.timestamps = []
        self.img_size = 512
        self.camera_intrinsics = None
        self.use_calibration = config["use_calib"]
        self.save_results = True

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        img = self.get_image(idx)
        timestamp = self.get_timestamp(idx)
        return timestamp, img

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        img = cv2.imread(str(self.rgb_files[idx]))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_image(self, idx):
        img = self.read_img(idx)
        if self.use_calibration:
            img = self.camera_intrinsics.remap(img)
        return img.astype(self.dtype) / 255.0

    def get_img_shape(self):
        img = self.read_img(0)
        raw_img_shape = img.shape
        img = resize_img(img, self.img_size)
        return img["img"][0].shape[1:], raw_img_shape[:2]

    def subsample(self, subsample):
        self.rgb_files = self.rgb_files[::subsample]
        self.timestamps = self.timestamps[::subsample]

    def has_calib(self):
        return self.camera_intrinsics is not None


class TUMDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        rgb_list = self.dataset_path / "rgb.txt"
        tstamp_rgb = np.loadtxt(rgb_list, delimiter=" ", dtype=np.unicode_, skiprows=0)
        self.rgb_files = [self.dataset_path / f for f in tstamp_rgb[:, 1]]
        self.timestamps = tstamp_rgb[:, 0]

        match = re.search(r"freiburg(\d+)", dataset_path)
        idx = int(match.group(1))
        if idx == 1:
            calib = np.array(
                [517.3, 516.5, 318.6, 255.3, 0.2624, -0.9531, -0.0054, 0.0026, 1.1633]
            )
        if idx == 2:
            calib = np.array(
                [520.9, 521.0, 325.1, 249.7, 0.2312, -0.7849, -0.0033, -0.0001, 0.9172]
            )
        if idx == 3:
            calib = np.array([535.4, 539.2, 320.1, 247.6])
        W, H = 640, 480
        self.camera_intrinsics = Intrinsics.from_calib(self.img_size, W, H, calib)


class EurocDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        # For Euroc dataset, the distortion is too much to handle for MASt3R.
        # So we always undistort the images, but the calibration will not be used
        # for any later optimization unless specified.
        self.use_calibration = True
        self.dataset_path = pathlib.Path(dataset_path)
        rgb_list = self.dataset_path / "mav0/cam0/data.csv"
        tstamp_rgb = np.loadtxt(rgb_list, delimiter=",", dtype=np.unicode_, skiprows=0)
        self.rgb_files = [
            self.dataset_path / "mav0/cam0/data" / f for f in tstamp_rgb[:, 1]
        ]
        self.timestamps = tstamp_rgb[:, 0]
        with open(self.dataset_path / "mav0/cam0/sensor.yaml") as f:
            self.cam0 = yaml.load(f, Loader=yaml.FullLoader)
        W, H = self.cam0["resolution"]
        intrinsics = self.cam0["intrinsics"]
        distortion = np.array(self.cam0["distortion_coefficients"])
        self.camera_intrinsics = Intrinsics.from_calib(
            self.img_size, W, H, [*intrinsics, *distortion], always_undistort=True
        )

    def read_img(self, idx):
        img = cv2.imread(str(self.rgb_files[idx]), cv2.IMREAD_GRAYSCALE)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class ETH3DDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        rgb_list = self.dataset_path / "rgb.txt"
        tstamp_rgb = np.loadtxt(rgb_list, delimiter=" ", dtype=np.unicode_, skiprows=0)
        self.rgb_files = [self.dataset_path / f for f in tstamp_rgb[:, 1]]
        self.timestamps = tstamp_rgb[:, 0]
        calibration = np.loadtxt(
            self.dataset_path / "calibration.txt",
            delimiter=" ",
            dtype=np.float32,
            skiprows=0,
        )
        _, (H, W) = self.get_img_shape()
        self.camera_intrinsics = Intrinsics.from_calib(self.img_size, W, H, calibration)


class SevenScenesDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        self.rgb_files = natsorted(
            list((self.dataset_path / "seq-01").glob("*.color.png"))
        )
        self.timestamps = np.arange(0, len(self.rgb_files)).astype(self.dtype)
        fx, fy, cx, cy = 585.0, 585.0, 320.0, 240.0
        self.camera_intrinsics = Intrinsics.from_calib(
            self.img_size, 640, 480, [fx, fy, cx, cy]
        )


class RealsenseDataset(MonocularDataset):
    def __init__(self):
        super().__init__()
        if not HAS_REALSENSE:
            raise ImportError("pyrealsense2 is required for RealsenseDataset")
        self.dataset_path = None
        self.pipeline = rs.pipeline()
        self.h, self.w = 480, 640
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.color, self.w, self.h, rs.format.bgr8, 30
        )
        self.profile = self.pipeline.start(self.rs_config)

        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        self.rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )
        self.save_results = False

        if self.use_calibration:
            rgb_intrinsics = self.rgb_profile.get_intrinsics()
            self.camera_intrinsics = Intrinsics.from_calib(
                self.img_size,
                self.w,
                self.h,
                [
                    rgb_intrinsics.fx,
                    rgb_intrinsics.fy,
                    rgb_intrinsics.ppx,
                    rgb_intrinsics.ppy,
                ],
            )

    def __len__(self):
        return 999999

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        frameset = self.pipeline.wait_for_frames()
        timestamp = frameset.get_timestamp()
        timestamp /= 1000
        self.timestamps.append(timestamp)

        rgb_frame = frameset.get_color_frame()
        img = np.asanyarray(rgb_frame.get_data())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(self.dtype)
        return img


class ColmapDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.use_calibration = True
        self.dataset_path = pathlib.Path(dataset_path)

        image_subdir = config.get("dataset", {}).get("image_subdir", "images")
        self.image_dir = self.dataset_path / image_subdir

        sparse_dir = self.dataset_path / config.get("dataset", {}).get("sparse_dir", "sparse/0")
        if not sparse_dir.exists():
            raise FileNotFoundError(f"No COLMAP sparse directory found at {sparse_dir}")

        cameras = read_cameras_binary(sparse_dir / "cameras.bin")
        images = read_images_binary(sparse_dir / "images.bin")

        cam = list(cameras.values())[0]
        model = cam.model

        if model == "PINHOLE":
            fx, fy, cx, cy = cam.params[:4]
            distortion = np.zeros(4)
        elif model == "SIMPLE_PINHOLE":
            fx = fy = cam.params[0]
            cx, cy = cam.params[1:3]
            distortion = np.zeros(4)
        elif model == "OPENCV":
            fx, fy, cx, cy = cam.params[:4]
            distortion = np.array(cam.params[4:8])
        elif model == "SIMPLE_RADIAL":
            fx = fy = cam.params[0]
            cx, cy = cam.params[1:3]
            distortion = np.zeros(4)
            distortion[0] = cam.params[3]
        elif model == "RADIAL":
            fx = fy = cam.params[0]
            cx, cy = cam.params[1:3]
            distortion = np.zeros(4)
            distortion[0:2] = cam.params[3:5]
        elif model == "OPENCV_FISHEYE":
            fx, fy, cx, cy = cam.params[:4]
            distortion = np.zeros(4)
            distortion[0:4] = cam.params[4:8]
        else:
            raise ValueError(f"Unsupported camera model: {model}")

        K = [fx, fy, cx, cy]
        W, H = cam.width, cam.height
        self.camera_intrinsics = Intrinsics.from_calib(self.img_size, W, H, [*K, *distortion])

        self.poses = {}
        for _, img in images.items():
            q = img.qvec
            t = img.tvec
            R = qvec2rotmat(q)
            T_CW = np.eye(4)
            T_CW[:3, :3] = R
            T_CW[:3, 3] = t
            T_WC = np.linalg.inv(T_CW)
            self.poses[img.name] = T_WC

        all_items = sorted(os.listdir(self.image_dir))
        self.rgb_files = [os.path.join(self.image_dir, item) for item in all_items]
        self.timestamps = np.array([i for i, _ in enumerate(self.rgb_files)], dtype=np.float32)
        self.rgb_poses = [self.poses.get(os.path.basename(p), np.eye(4)) for p in self.rgb_files]

    def get_pose(self, index):
        return self.rgb_poses[index]

    def get_image_name(self, index):
        return os.path.basename(self.rgb_files[index])


class Webcam(MonocularDataset):
    def __init__(self):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = None
        self.cap = cv2.VideoCapture(-1)
        self.save_results = False

    def __len__(self):
        return 999999

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        ret, img = self.cap.read()
        if not ret:
            raise ValueError("Failed to read image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.timestamps.append(idx / 30)
        return img


class MP4Dataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = pathlib.Path(dataset_path)
        if HAS_TORCHCODEC:
            self.decoder = VideoDecoder(str(self.dataset_path))
            self.fps = self.decoder.metadata.average_fps
            self.total_frames = self.decoder.metadata.num_frames
        else:
            print("torchcodec is not installed. This may slow down the dataloader")
            self.cap = cv2.VideoCapture(str(self.dataset_path))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.stride = config["dataset"]["subsample"]

    def __len__(self):
        return self.total_frames // self.stride

    def read_img(self, idx):
        if HAS_TORCHCODEC:
            img = self.decoder[idx * self.stride]
            img = img.permute(1, 2, 0)
            img = img.numpy()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx * self.stride)
            ret, img = self.cap.read()
            if not ret:
                raise ValueError("Failed to read image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(self.dtype)
        timestamp = idx / self.fps
        self.timestamps.append(timestamp)
        return img


class RGBFiles(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = pathlib.Path(dataset_path)
        self.rgb_files = natsorted(list(self.dataset_path.glob("*.png")))
        if not self.rgb_files:
            self.rgb_files = natsorted(list(self.dataset_path.glob("*.jpg")))
        self.timestamps = np.arange(0, len(self.rgb_files)).astype(self.dtype) / 30.0


class Intrinsics:
    def __init__(self, img_size, W, H, K_orig, K, distortion, mapx, mapy):
        self.img_size = img_size
        self.W, self.H = W, H
        self.K_orig = K_orig
        self.K = K
        self.distortion = distortion
        self.mapx = mapx
        self.mapy = mapy
        _, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
            np.zeros((H, W, 3)), self.img_size, return_transformation=True
        )
        self.K_frame = self.K.copy()
        self.K_frame[0, 0] = self.K[0, 0] / scale_w
        self.K_frame[1, 1] = self.K[1, 1] / scale_h
        self.K_frame[0, 2] = self.K[0, 2] / scale_w - half_crop_w
        self.K_frame[1, 2] = self.K[1, 2] / scale_h - half_crop_h

    def remap(self, img):
        return cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)

    @staticmethod
    def from_calib(img_size, W, H, calib, always_undistort=False):
        if not config["use_calib"] and not always_undistort:
            return None
        fx, fy, cx, cy = calib[:4]
        distortion = np.zeros(4)
        if len(calib) > 4:
            distortion = np.array(calib[4:])
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        K_opt = K.copy()
        mapx, mapy = None, None
        center = config["dataset"]["center_principle_point"]
        K_opt, _ = cv2.getOptimalNewCameraMatrix(
            K, distortion, (W, H), 0, (W, H), centerPrincipalPoint=center
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            K, distortion, None, K_opt, (W, H), cv2.CV_32FC1
        )

        return Intrinsics(img_size, W, H, K, K_opt, distortion, mapx, mapy)


# Dataset type registry
_DATASET_TYPES = {
    "tum": TUMDataset,
    "euroc": EurocDataset,
    "eth3d": ETH3DDataset,
    "7-scenes": SevenScenesDataset,
    "realsense": lambda _: RealsenseDataset(),
    "webcam": lambda _: Webcam(),
    "mp4": MP4Dataset,
    "rgbfiles": RGBFiles,
    "colmap": ColmapDataset,
}


def load_dataset(dataset_path, dataset_type="auto"):
    """Load dataset by explicit type or auto-detection from path."""

    # Explicit type
    if dataset_type != "auto":
        if dataset_type in _DATASET_TYPES:
            return _DATASET_TYPES[dataset_type](dataset_path)
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Auto-detection
    path_str = str(dataset_path).lower()

    for keyword in ("tum", "euroc", "eth3d", "7-scenes"):
        if keyword in path_str:
            return _DATASET_TYPES[keyword](dataset_path)

    if path_str.endswith((".mp4", ".avi", ".mov")):
        return MP4Dataset(dataset_path)

    if os.path.exists(os.path.join(dataset_path, "sparse/0/images.bin")):
        return ColmapDataset(dataset_path)

    img_count = len([f for f in os.listdir(dataset_path)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if img_count > 10:
        return RGBFiles(dataset_path)

    raise ValueError(f"Cannot infer dataset type from path: {dataset_path}")
