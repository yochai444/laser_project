"""
V3 model: STFT-mask U-Net.

Input : noisy microphone waveform
Output: enhanced microphone waveform

The network predicts a soft time-frequency mask in [0,1] from the noisy
microphone magnitude. The original noisy microphone phase is retained.
The laser is never an inference input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.GELU(),
    )


class MaskUNet2D(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.e1 = conv_block(1, base)
        self.e2 = conv_block(base, base * 2)
        self.e3 = conv_block(base * 2, base * 4)
        self.e4 = conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.b = conv_block(base * 8, base * 16)

        self.d4 = conv_block(base * 16 + base * 8, base * 8)
        self.d3 = conv_block(base * 8 + base * 4, base * 4)
        self.d2 = conv_block(base * 4 + base * 2, base * 2)
        self.d1 = conv_block(base * 2 + base, base)

        self.head = nn.Conv2d(base, 1, 1)

    @staticmethod
    def _up_to(x, ref):
        return F.interpolate(
            x, size=ref.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))

        d4 = self.d4(torch.cat([self._up_to(b, e4), e4], dim=1))
        d3 = self.d3(torch.cat([self._up_to(d4, e3), e3], dim=1))
        d2 = self.d2(torch.cat([self._up_to(d3, e2), e2], dim=1))
        d1 = self.d1(torch.cat([self._up_to(d2, e1), e1], dim=1))

        return torch.sigmoid(self.head(d1))


class OpticalGuidedEnhancer(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop=128,
        win=512,
        base=24,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.win = int(win)
        self.net = MaskUNet2D(base=base)
        self.register_buffer("window", torch.hann_window(self.win))

    def stft(self, wav):
        return torch.stft(
            wav.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window,
            center=True,
            return_complex=True,
        )

    def istft(self, spec, length):
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window,
            center=True,
            length=length,
        ).unsqueeze(1)

    def forward(self, noisy):
        length = noisy.shape[-1]
        X = self.stft(noisy)
        mag = X.abs()

        # Per-example log-magnitude normalization for a stable network input.
        feat = torch.log1p(mag)
        mu = feat.mean(dim=(-2, -1), keepdim=True)
        sd = feat.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        feat = (feat - mu) / sd

        mask = self.net(feat.unsqueeze(1)).squeeze(1)
        Y = X * mask
        enhanced = self.istft(Y, length)

        return {
            "enhanced": enhanced,
            "mask": mask,
            "noisy_spec": X,
            "enhanced_spec": Y,
        }
