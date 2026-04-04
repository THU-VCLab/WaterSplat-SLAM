<h2 align="center">
WaterSplat-SLAM: Photorealistic Monocular SLAM in Underwater Environment
</h2>

<h3 align="center">
IEEE Robotics and Automation Letters (RA-L), 2026
</h3>

<p align="center">
Kangxu Wang<sup>*</sup> · 
Shaofeng Zou<sup>*</sup> · 
Chenxing Jiang<sup>*</sup> · 
Yixiang Dai · 
Siang Chen · 
Shaojie Shen · 
Guijin Wang<sup>†</sup>
</p>

<!-- <h3 align="center">
    <a href="https://arxiv.org/pdf/####">Paper</a> | <a href="#">Project Page</a>
</h3> -->

<div align="center">
    <img alt="WaterSplat-SLAM" src=".assets/pipeline.png" />
</div>

<p align="justify"> Underwater monocular SLAM is a highly challenging problem with applications ranging from autonomous underwater vehicles to marine archaeology. However, existing underwater SLAM methods struggle to generate high-fidelity rendered maps. We propose WaterSplat-SLAM, the first novel monocular underwater SLAM system to achieve robust pose estimation and photorealistic dense map construction to our knowledge.
Specifically, we combine semantic medium filtering with a dual-view 3D reconstruction prior to achieve underwater adaptive camera tracking and depth estimation. Furthermore, we propose a semantically guided rendering and adaptive map management strategy, combined with an online medium-aware Gaussian map, to model the underwater environment in a photorealistic and compact manner. Experiments on multiple underwater datasets demonstrate that WaterSplat-SLAM achieves robust camera tracking and high-fidelity rendering in underwater environments.
</p>

## Installation

### Prerequisites

- Linux (tested on Ubuntu 20.04)
- Python 3.10
- CUDA 11.8+ (tested with CUDA 11.8)
- GPU with at least 8GB VRAM (12GB+ recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/THU-VCLab/WaterSplat-SLAM.git --recursive
cd WaterSplat-SLAM
```

If you already cloned without `--recursive`, initialize the submodules:
```bash
git submodule update --init --recursive
```

### Step 2: Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate WaterSplat-SLAM
```

### Step 3: Install PyTorch

Install PyTorch with the appropriate CUDA version:

```bash
# CUDA 11.8 (default, recommended)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### Step 4: Install Third-party Dependencies

```bash
# Install MASt3R (3D reconstruction backbone)
pip install -e thirdparty/mast3r

# Install in3d (visualization utilities)
pip install -e thirdparty/in3d

# Install fused-ssim (CUDA-accelerated SSIM loss)
pip install --no-build-isolation thirdparty/fused-ssim

# Install lietorch (Lie group operations)
pip install lietorch@git+https://github.com/princeton-vl/lietorch.git
```

### Step 5: Build SLAM Backend (CUDA extensions)

```bash
pip install --no-build-isolation -e .
```

This compiles the Gauss-Newton solver CUDA kernels for the SLAM backend.

### Step 6: Build Water Gaussian Renderer

```bash
cd water_gaussian
pip install --no-build-isolation -e .
cd ..
```

This compiles the `cudalight` CUDA module for underwater-adapted Gaussian rendering.

### Step 7: Install tiny-cuda-nn

Required for the medium MLP network encoding:

```bash
pip install ninja
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn
cd tiny-cuda-nn
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
cd bindings/torch
python setup.py install
cd ../../..
```

### Step 8: Download Checkpoints

Download the MASt3R pretrained weights and retrieval codebook:

```bash
mkdir -p checkpoints/
# MASt3R model weights
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
# Retrieval model weights
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
# Retrieval codebook
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

Download the CLIPSeg weight for water segmentation:

```bash
mkdir -p weights/
wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights/rd64-uni.pth
```

## Usage

### Running on Underwater Datasets

```bash
python main.py --dataset <path_to_dataset> --config config/Panama.yaml
```

**Command-line arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Path to the dataset directory (or video file) | — |
| `--config` | Path to config YAML file | `config/base.yaml` |
| `--save-as` | Name for saving results | `default` |
| `--device`       | CUDA device                                   | `cuda:0`           |
| `--dataset_type` | Dataset format                                | `auto`             |

**Supported dataset formats:**

| Type | Description | Auto-detected |
|------|-------------|---------------|
| `colmap` | COLMAP sparse reconstruction with images | Yes (if `sparse/0/images.bin` exists) |
| `tum` | TUM RGB-D format (`rgb.txt` + images) | Yes (if path contains "tum") |
| `euroc` | EuRoC MAV format | Yes (if path contains "euroc") |
| `eth3d` | ETH3D format | Yes (if path contains "eth3d") |
| `7-scenes` | Microsoft 7-Scenes | Yes (if path contains "7-scenes") |
| `mp4` | Video file (`.mp4`, `.avi`, `.mov`) | Yes (by file extension) |
| `rgbfiles` | Directory of PNG/JPG images (no calibration) | Yes (fallback if >10 images found) |
| `realsense` | Intel RealSense live camera | No (must specify `--dataset_type realsense`) |
| `webcam` | USB webcam live capture | No (must specify `--dataset_type webcam`) |

When `--dataset_type` is omitted or set to `auto`, the system detects the format from the dataset path. For datasets with custom intrinsics (e.g., ROV cameras), use `--dataset_type rgbfiles` with `--calib` pointing to a calibration YAML file.

### Example: SeathruNeRF Underwater Datasets

```bash
# Panama
python main.py --dataset /path/to/SeathruNeRF/Panama --config config/Panama.yaml
# Curasao
python main.py --dataset /path/to/SeathruNeRF/Curasao --config config/Curasao.yaml
# JapaneseGardens-RedSea
python main.py --dataset /path/to/SeathruNeRF/JapaneseGardens --config config/Jap_RedSea.yaml
# IUI3-RedSea
python main.py --dataset /path/to/SeathruNeRF/IUI3 --config config/RedSea.yaml
```

### Example: WaterSplat-SLAM Datasets

```bash
python main.py --dataset /path/to/pool_up --config config/pool_up.yaml
python main.py --dataset /path/to/pool_up2 --config config/pool_up2.yaml
python main.py --dataset /path/to/5_pool --config config/5_pool.yaml
python main.py --dataset /path/to/pipe_local --config config/pipe_local.yaml
python main.py --dataset /path/to/big_gate --config config/big_gate.yaml
```

### Example: Custom Underwater Data

**Option 1: COLMAP format (recommended, with calibration)**

```
your_dataset/
├── images/          # RGB images
└── sparse/
    └── 0/           # COLMAP sparse reconstruction
        ├── cameras.bin (or cameras.txt)
        ├── images.bin (or images.txt)
        └── points3D.bin (or points3D.txt)
```

```bash
python main.py --dataset /path/to/your_dataset --config config/base.yaml
```

**Option 2: Image folder with external calibration**

For cameras with known intrinsics (e.g., ROV, GoPro), place images in a folder and provide a calibration YAML:

```yaml
# my_calib.yaml
width: 1920
height: 1080
calibration: [fx, fy, cx, cy]  # or [fx, fy, cx, cy, k1, k2, p1, p2] with distortion
```

```bash
python main.py --dataset /path/to/image_folder --calib my_calib.yaml --dataset_type rgbfiles
```

**Option 3: Video file**

```bash
python main.py --dataset /path/to/video.mp4 --config config/base.yaml
```

Note: Video input runs without camera calibration by default. For better accuracy, provide calibration via `--calib`.

### Output

Results are saved to `output.base_dir` specified in config:
## Acknowledgements

This project builds upon several excellent works:
- [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM) - Dense stereo SLAM backbone
- [MASt3R](https://github.com/naver/mast3r) - 3D reconstruction from image pairs
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Neural radiance field rendering
- [Water-Splatting](https://github.com/water-splatting/water-splatting) - Underwater Gaussian splatting with medium rendering
- [Lowlight Splatting Underwater](https://github.com/booqo/lowlight-underwater-for-nerf-studio) - Underwater cudalight renderer
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) - CUDA backend kernels
- [gsplat](https://github.com/nerfstudio-project/gsplat) - Gaussian splatting library

## License

This project is licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License. See [LICENSE.md](LICENSE.md) for details.

## Citation

If you find this work useful, please cite:

```bibtex
@ARTICLE{11417448,
  author={Wang, Kangxu and Zou, Shaofeng and Jiang, Chenxing and Dai, Yixiang and Chen, Siang and Shen, Shaojie and Wang, Guijin},
  journal={IEEE Robotics and Automation Letters}, 
  title={WaterSplat-SLAM: Photorealistic Monocular SLAM in Underwater Environment}, 
  year={2026},
  volume={11},
  number={5},
  pages={5614-5621},
  doi={10.1109/LRA.2026.3668465}
}
```
