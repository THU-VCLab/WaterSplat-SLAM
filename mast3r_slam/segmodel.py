import torch
import requests

from mast3r_slam.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

import torch
import requests
from PIL import Image
import torch.nn.functional as F
from mast3r_slam.clipseg import CLIPDensePredT
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import time



class WaterSegmenter:
    def __init__(self, model_path='weights/rd64-uni.pth', device='cuda'):
        # 初始化模型[1](@ref)
        self.device = torch.device(device)
        self.model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64).to(self.device)
        self.model.eval()

        # 加载预训练权重（仅Decoder部分）[1](@ref)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict, strict=False)

        #图像预处理流程[1](@ref)
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((320, 320))
        ])


    def predict_mask(self, image_input, prompt):
        if isinstance(image_input, str):
            if image_input.startswith('http'):
                img = Image.open(requests.get(image_input, stream=True).raw)
            else:
                img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise ValueError("输入类型需为路径/PIL图像/URL")

        # 记录原始尺寸用于结果还原[1](@ref)
        original_size = img.size  # (width, height)

    
        img_tensor = self.transform2(img).unsqueeze(0).to(self.device)
        # 模型推理[1](@ref)
        with torch.no_grad():
            pred = self.model(img_tensor, [prompt])[0]
            prob_mask = torch.sigmoid(pred[0][0])  # 概率图

        # 后处理
        mask = (prob_mask > 0.7).float()  # 二值化阈值
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=original_size[::-1],  # 目标尺寸(height, width)
            mode='bilinear'
        ).squeeze()

        return mask
