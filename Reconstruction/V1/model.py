import torch
import torch.nn as nn
import torch.nn.functional as F

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, x, skip_x):
        x = self.up(x)
        
        # התאמת מידות דינמית כדי למנוע קריסות
        diff = skip_x.size(-1) - x.size(-1)
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[:, :, :skip_x.size(-1)]
            
        x = x + skip_x
        return self.relu(self.norm(x))

class SimpleAudioUNet(nn.Module):
    def __init__(self, in_channels=2): 
        super().__init__()
        self.down1 = DownConv(in_channels, 32)
        self.down2 = DownConv(32, 64)
        self.down3 = DownConv(64, 128)
        
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=15, padding=7),
            nn.LeakyReLU(0.2)
        )
        
        self.up1 = UpConv(128, 64)
        self.up2 = UpConv(64, 32)
        
        self.out_conv = nn.ConvTranspose1d(32, 1, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        bn = self.bottleneck(d3)
        u1 = self.up1(bn, d2)
        u2 = self.up2(u1, d1)
        out = self.out_conv(u2)
        
        # הבטחה שהפלט הסופי יהיה בדיוק באורך של הקלט
        diff = x.size(-1) - out.size(-1)
        if diff > 0:
            out = F.pad(out, (0, diff))
        elif diff < 0:
            out = out[:, :, :x.size(-1)]
            
        return self.tanh(out)