# WaterSplat-SLAM: Photorealistic Monocular SLAM in Underwater Environment
<p align="center">
Kangxu Wang,
Shaofeng Zou
</p>

<h3 align="center">
    <a href="https://arxiv.org/pdf/####">📄 Paper</a> | <a href="#">🌐 Project Page</a>
</h3>

<div align="center">
    <img alt="WaterSplat-SLAM" src=".assets/pipeline.jpeg" />
</div>

<p align="justify"> We propose WaterSplat-SLAM, to our knowledge, the first photorealistic monocular underwater SLAM system that achieves robust pose estimation and photorealistic dense mapping. We design a novel underwater adapted 3DGS SLAM framework that combines a two-view geometry module for generating multi-view consistent depths and globally consistent poses, and a Gaussian map with medium rendering for photorealistic underwater mapping. We further semantically segment pure-water regions from image, which both suppresses their effects on camera tracking and guides rendering to avoid inaccurately representing volumetric properties with Gaussian primitives. To maintain global consistency and reduce redundancy, Gaussian primitives are adjusted and merged upon loop closure. Experiments on multiple underwater datasets demonstrate that WaterSplat-SLAM achieves robust camera tracking, high-fidelity rendering, and detailed reconstruction in underwater environments.
<br>

## Installation
### Prerequisites

- Python 3.8+
- CUDA 11.8+

### 1. Create Environment
```bash
git clone git@github.com:knightzzz9w/WaterSplatting-SLAM.git --recursive
```

if you've clone the repo without --recursive run
```bash
git submodule update --init --recursive
```
Create and activate a Conda environment:
```bash
conda env create -f environment.yml
conda activate WaterSplat-SLAM
nvcc --version
```

### 2. Install PyTorch
The default is CUDA 11.8. If you use other versions, you need to modify the cuda part in `environment.yaml`.

```bash
# CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```
### 3. Install Thirdparty Dependence
```bash
pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install -e thirdparty/simple-knn
pip install --no-build-isolation -e .
```

### 4. Install Gaussian Backen
```bash
cd water_gaussian
pip install -e . # install cudalight in local
```
### 5. Install tinycudann

```bash
pip install ninja
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn
cd tiny-cuda-nn
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
cd bindings/torch
python setup.py install
sudo apt updata
```