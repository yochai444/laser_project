"""
Stage 3 — model.

Changes from the previous version, and why:

  depth       3 encoder levels gave a receptive field of ~211 samples. At 16 kHz that
              is 13 ms — less than a single phoneme, and far too little context to
              separate speech from noise. This version uses 5 levels plus a recurrent
              bottleneck. At 4 kHz the convolutional receptive field alone is ~0.9 s,
              and the BLSTM sees the whole chunk.

  upsampling  ConvTranspose1d(kernel_size=15, stride=2) produces checkerboard
              artifacts because the kernel size is not divisible by the stride; three
              such layers in series is what the periodic striping in the output
              spectrogram most likely came from. Replaced with nearest-neighbour
              upsampling followed by a plain convolution, which cannot alias in that
              way.

  norm        BatchNorm1d with batch_size=4 estimates its statistics from four
              examples per step. GroupNorm is independent of batch size.

  skips       Concatenation rather than addition, so the decoder can weight encoder
              detail rather than being forced to sum it in.

  output      No tanh. The target is RMS-normalised, not peak-normalised, so it is
              not bounded to [-1, 1] and a tanh would clip the loudest samples.
"""

import torch
import torch.nn as nn


def norm(ch):
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class EncoderBlock(nn.Module):
    def __init__(self, cin, cout, kernel=15, stride=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, kernel, stride=stride, padding=kernel // 2),
            norm(cout),
            nn.GELU(),
            nn.Conv1d(cout, cout, 1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, cin, cskip, cout, kernel=15):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.net = nn.Sequential(
            nn.Conv1d(cin + cskip, cout, kernel, padding=kernel // 2),
            norm(cout),
            nn.GELU(),
            nn.Conv1d(cout, cout, 1),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            n = min(x.shape[-1], skip.shape[-1])
            x, skip = x[..., :n], skip[..., :n]
        return self.net(torch.cat([x, skip], dim=1))


class OpticalUNet(nn.Module):
    """
    Maps a single-channel microphone waveform to the optically reconstructed
    waveform. One input channel: the laser is not required at inference.
    """

    def __init__(self, in_channels=1, base=32, depth=5, lstm_layers=2):
        super().__init__()
        self.depth = depth

        chans = [base * (2 ** i) for i in range(depth)]      # 32 64 128 256 512
        self.encoders = nn.ModuleList()
        cin = in_channels
        for c in chans:
            self.encoders.append(EncoderBlock(cin, c))
            cin = c

        self.lstm = nn.LSTM(
            input_size=chans[-1],
            hidden_size=chans[-1] // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_proj = nn.Conv1d(chans[-1], chans[-1], 1)

        self.decoders = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.decoders.append(DecoderBlock(chans[i], chans[i - 1], chans[i - 1]))

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(chans[0], chans[0], 15, padding=7),
            nn.GELU(),
            nn.Conv1d(chans[0], 1, 1),
        )

    def forward(self, x):
        n_in = x.shape[-1]
        pad = (-n_in) % (2 ** self.depth)
        if pad:
            x = nn.functional.pad(x, (0, pad))

        skips = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)

        seq = h.transpose(1, 2)
        seq, _ = self.lstm(seq)
        h = h + self.lstm_proj(seq.transpose(1, 2))

        for dec, skip in zip(self.decoders, reversed(skips[:-1])):
            h = dec(h, skip)

        out = self.head(h)
        return out[..., :n_in]


if __name__ == "__main__":
    m = OpticalUNet()
    n = sum(p.numel() for p in m.parameters())
    y = m(torch.randn(2, 1, 8000))
    print(f"params: {n/1e6:.2f}M   in (2,1,8000) -> out {tuple(y.shape)}")
