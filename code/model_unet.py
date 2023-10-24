import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(Unet, self).__init__()
        ################# (batch, input channel, w, h)      ->  (batch, 64, w, h)
        self.conv_block = Conv_Block(input_channel, 64)

        ################# (batch, 64, w, h)                 ->  (batch, 128, w/2, h/2)                      
        self.down_block_1 = nn.Sequential(  
            nn.MaxPool2d(2),
            Conv_Block(64, 128))
        ################# (batch, 128, w/2, h/2)            ->  (batch, 256, w/4, h/4)
        self.down_block_2 = nn.Sequential(
            nn.MaxPool2d(2),
            Conv_Block(128, 256) )
        ################# (batch, 256, w/4, h/4)            ->  (batch, 512, w/8, w/8)
        self.down_block_3 = nn.Sequential(
            nn.MaxPool2d(2),
            Conv_Block(256, 512) )
        ################# (batch, 512, w/8, h/8)            ->  (batch, 1024, w/16, h/16)
        self.down_block_4 = nn.Sequential(
            nn.MaxPool2d(2),
            Conv_Block(512, 512) )

        ################# (batch, 1024, w/16, h/16)         ->  (batch, 512, w/8, h/8)
        self.upscale_block_1 = Upscale_Block(1024, 512 // 2)
        ################# (batch, 512, w/8, h/8)            ->  (batch, 256, w/4, h/4)
        self.upscale_block_2 = Upscale_Block(512, 256 // 2)
        ################# (batch, 256, w/4, h/4)            ->  (batch, 128, w/2, h/2)
        self.upscale_block_3 = Upscale_Block(256, 128 // 2)
        ################# (batch, 128, w/2, h/2)            ->  (batch, 64, w, h)
        self.upscale_block_4 = Upscale_Block(128, 64)


        ################# (batch, 64, w, h)                 ->  (batch, 3, w, h)
        self.out_conv = nn.Conv2d(64, out_channel, kernel_size=1)



    def forward(self, x):                   # (batch, input channel, w, h)
        x1 = self.conv_block(x)             # (batch, 64, w, h)

        x2 = self.down_block_1(x1)          # (batch, 128, w/2, h/2)
        x3 = self.down_block_2(x2)          # (batch, 256, w/4, h/4)
        x4 = self.down_block_3(x3)          # (batch, 512, w/8, w/8)
        x5 = self.down_block_4(x4)          # (batch, 512, w/16, h/16)

        x = self.upscale_block_1(x5, x4)    # (batch, 256, w/8, w/8)
        x = self.upscale_block_2(x , x3)    # (batch, 128, w/4, h/4)
        x = self.upscale_block_3(x , x2)    # (batch, 64, w/2, h/2)
        x = self.upscale_block_4(x , x1)    # (batch, 64, w, h)
        return self.out_conv(x)



class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv_block(x)


class Upscale_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_block = Conv_Block(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)