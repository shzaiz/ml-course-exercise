from typing import Any
import torch


from torch import nn
class AlexNet(nn.Module):
    def __init__(self, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.features = nn.Sequential(
            # The first convolutional layer filters the 224×224×3 input image 
            # with 96 kernels of size 11×11×3 with a stride of 4 pixels
            nn.Conv2d(in_channels=1, out_channels=96, stride=4, kernel_size=11), # 55x55x96
            #         ^~~~~~~~~~~~~ Change here because MNIST has only 1 dim
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 27 x 27 x 96

            # The second convolutional layer takes as input the (response-normalized
            # and pooled) output of the first convolutional layer and filters 
            # it with 256 kernels of size 5 × 5 × 48.
            nn.Conv2d(in_channels=96, out_channels=256, padding=2, kernel_size=5), # 27 x 27 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 13 x 13 x 256

            # The third convolutional layer has 384 kernels of size 3 × 3 ×
            # 256 connected to the (normalized, pooled) outputs of the second 
            # convolutional layer.
            nn.Conv2d(in_channels=256, out_channels=384, padding=1, kernel_size=3), # 13 x 13 x 384
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, padding=1, kernel_size=3), # 13 x 13 x 384
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, padding=1, kernel_size=3), # 13 x 13 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2) # 6 x 6 x 256
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size=1*227*227, hidden1=512, hidden2=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                            # 扁平化至 1×(227*227)
            nn.Linear(input_size, hidden1),          # 第一隐藏层
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                         # 可选 Dropout
            nn.Linear(hidden1, hidden2),             # 第二隐藏层
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden2, num_classes)          # 输出层
        )
    def forward(self, x):
        return self.net(x)
