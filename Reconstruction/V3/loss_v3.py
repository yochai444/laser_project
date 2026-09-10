"""
V3 loss.

Primary objective:
    reconstruct the CLEAN MICROPHONE.

Auxiliary optical objective:
    preserve the temporal speech-activity pattern visible in the synchronized laser.

The optical term is intentionally envelope-based rather than waveform L1 because the
two sensors have different transfer functions and phase structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def si_sdr(pred, target, eps=1e-8):
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    alpha = (pred * target).sum(-1, keepdim=True) / (
        target.pow(2).sum(-1, keepdim=True) + eps
    )
    proj = alpha * target
    noise = pred - proj
    return 10 * torch.log10(
        proj.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps) + eps
    )


def pearson_flat(a, b, eps=1e-8):
    a = a.flatten(1)
    b = b.flatten(1)
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    den = torch.sqrt(
        (a.square().sum(1) * b.square().sum(1)).clamp_min(eps)
    )
    return (a * b).sum(1) / den


class V3Loss(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        hop=128,
        win=512,
        band=(150.0, 1400.0),
        lambda_wave=0.5,
        lambda_spec=1.0,
        lambda_mask=0.5,
        lambda_optical=0.25,
    ):
        super().__init__()
        self.sr = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.win = int(win)
        self.lambda_wave = float(lambda_wave)
        self.lambda_spec = float(lambda_spec)
        self.lambda_mask = float(lambda_mask)
        self.lambda_optical = float(lambda_optical)

        self.register_buffer("window", torch.hann_window(self.win))
        freqs = torch.linspace(0, self.sr / 2, self.n_fft // 2 + 1)
        self.register_buffer(
            "opt_band",
            (freqs >= band[0]) & (freqs <= min(band[1], self.sr / 2)),
        )

    def _stft(self, x):
        return torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window,
            center=True,
            return_complex=True,
        )

    def _optical_activity(self, optical):
        L = self._stft(optical).abs()[:, self.opt_band, :]
        e = torch.log1p(L).mean(dim=1)
        # Normalize each example across time to remove arbitrary optical gain.
        lo = e.amin(dim=1, keepdim=True)
        hi = e.amax(dim=1, keepdim=True)
        return (e - lo) / (hi - lo + 1e-6)

    @staticmethod
    def _pred_activity_from_mask(mask):
        # Mean mask across frequency is a soft "speech kept" activity trace.
        e = mask.mean(dim=1)
        lo = e.amin(dim=1, keepdim=True)
        hi = e.amax(dim=1, keepdim=True)
        return (e - lo) / (hi - lo + 1e-6)

    def forward(self, out, noisy, clean, optical, optical_weight=None):
        enhanced = out["enhanced"]
        mask = out["mask"]
        X = out["noisy_spec"]

        S = self._stft(clean)
        Y = out["enhanced_spec"]

        # 1) Clean microphone waveform reconstruction.
        wave = (enhanced - clean).abs().mean()

        # 2) Clean microphone spectral magnitude reconstruction.
        spec = (
            torch.log1p(Y.abs()) - torch.log1p(S.abs())
        ).abs().mean()

        # 3) Ideal-ratio-mask supervision is available because noisy speech
        #    was synthetically generated from the clean microphone.
        noise_mag = (X - S).abs()
        irm = S.abs() / (S.abs() + noise_mag + 1e-6)
        mask_loss = (mask - irm.detach()).abs().mean()

        # 4) Optical auxiliary supervision: temporal speech activity only.
        opt_act = self._optical_activity(optical)
        pred_act = self._pred_activity_from_mask(mask)

        # STFT frame counts should match, but crop defensively.
        t = min(opt_act.shape[-1], pred_act.shape[-1])
        opt_act = opt_act[:, :t]
        pred_act = pred_act[:, :t]
        per_ex_opt = (pred_act - opt_act).abs().mean(dim=1)

        if optical_weight is not None:
            w = optical_weight.to(per_ex_opt.device).clamp(0.0, 1.0)
            if w.sum() > 1e-6:
                optical_loss = (per_ex_opt * w).sum() / w.sum()
            else:
                optical_loss = per_ex_opt.mean() * 0.0
        else:
            optical_loss = per_ex_opt.mean()

        total = (
            self.lambda_wave * wave
            + self.lambda_spec * spec
            + self.lambda_mask * mask_loss
            + self.lambda_optical * optical_loss
        )

        return {
            "total": total,
            "wave": wave.detach(),
            "spec": spec.detach(),
            "mask": mask_loss.detach(),
            "optical": optical_loss.detach(),
        }
