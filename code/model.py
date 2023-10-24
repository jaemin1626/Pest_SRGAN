import math
import torch
from torch import batch_norm, nn
from torchvision.models import resnet101

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()
        
        upsample_block_num = int(math.log(scale_factor, 2))

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)
        
        #self.attention = Attention()
        # self.upsample = UpsampleBLock(64,2)
        # self.last_conv = nn.Conv2d(64,3,kernel_size=9,padding=4)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        x7 = self.block7(x6)
        x8 = self.block8(x7)

        return x7,(torch.tanh(x8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )
        ''' # 디버깅 용도
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bnrm1 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bnrm2 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bnrm3 = nn.BatchNorm2d(128)
        self.relu4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bnrm4 = nn.BatchNorm2d(256)
        self.relu5 = nn.LeakyReLU(0.2)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bnrm5 = nn.BatchNorm2d(256)
        self.relu6 = nn.LeakyReLU(0.2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bnrm6 = nn.BatchNorm2d(512)
        self.relu7 = nn.LeakyReLU(0.2)

        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bnrm7 = nn.BatchNorm2d(512)
        self.relu8 = nn.LeakyReLU(0.2)

        self.Apool = nn.AdaptiveAvgPool2d(1)
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=1)
        self.relu9 = nn.LeakyReLU(0.2)
        self.conv10= nn.Conv2d(1024, 1, kernel_size=1)
        '''

    def forward(self, x):
        batch_size = x.size(0)
        ''' # 디버깅 용도
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bnrm1(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bnrm2(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bnrm3(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bnrm4(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bnrm5(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.bnrm6(x)
        x = self.relu7(x)

        x = self.conv8(x)
        x = self.bnrm7(x)
        x = self.relu8(x)

        x = self.Apool(x)
        x = self.conv9(x)
        x = self.relu9(x)
        x = self.conv10(x)

        return torch.sigmoid(x.view(batch_size))
        '''
        return torch.sigmoid(self.net(x).view(batch_size))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x0