import torch
import matplotlib.pyplot as plt
from typing import Any
from torch import nn
from model import AlexNet
from matplotlib import colormaps

model = AlexNet(num_classes=10)
state = torch.load('alexnet_mnist.pth', map_location='cpu')
model.load_state_dict(state)
model.eval()

# 提取所有卷积层并可视化每层前8个滤波器（第一输入通道）
conv_layers = [layer for layer in model.features if isinstance(layer, nn.Conv2d)]
for idx, conv in enumerate(conv_layers, start=1):
    weights = conv.weight.data  # shape [out_ch, in_ch, k_h, k_w]
    # 取第一输入通道的前8个滤波器
    kernels = weights[:, 0, :, :][:8]

    plt.figure(figsize=(8, 4))
    for i, kern in enumerate(kernels):
        plt.subplot(2, 4, i + 1)
        plt.imshow(kern, aspect='equal', cmap=plt.get_cmap('Greys'))
        plt.axis('off')
    plt.suptitle(f'Conv Layer {idx}\'s first 8 filters (in_ch=0)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()