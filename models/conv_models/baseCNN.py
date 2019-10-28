import torch.nn as nn

# Conv2d에 BatchNord2d & Activate까지 해주는 레이어 합
class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias = False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.activate = nn.Hardtanh(0, 20, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activate(x)

class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv = nn.Sequential(
            BasicConv2d(1, 16),
            BasicConv2d(16, 32),
            BasicConv2d(32, 64),
            nn.MaxPool2d(2, 2),
            BasicConv2d(64, 128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)