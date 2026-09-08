"""
Stage 4 — loss.

Changes from the previous version, and why:

  band        The loss is now restricted to the band where the two channels actually
              agree (default 150-1400 Hz). Outside it the optical target contains
              either drift or nothing, so gradients from those bins are noise. The
              old loss ran over all 513 bins of a 1024-point STFT at 16 kHz, i.e. up
              to 8 kHz, where the target is silent by construction.

  resolutions Three STFT resolutions instead of one. A single 1024-point window at a
              256 hop resolves either time or frequency badly; the standard fix in the
              waveform-domain literature is to sum the loss over several.

  balance     Spectral convergence uses a relative Frobenius norm and is therefore
              dominated by the highest-energy bins. Pairing it with a log-magnitude
              L1 term, which weights every bin equally, keeps mid-band structure from
              being ignored. Both were present before, but over the wrong band.

  weighting   Optional per-example weights so low-coherence pairs contribute less.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BandSTFTLoss(nn.Module):
    def __init__(self, n_fft, hop, win, sample_rate, band=(150.0, 1400.0)):
        super().__init__()
        self.n_fft, self.hop, self.win_len = n_fft, hop, win
        self.register_buffer("window", torch.hann_window(win))
        freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        mask = (freqs >= band[0]) & (freqs <= min(band[1], sample_rate / 2))
        if not mask.any():
            mask[:] = True
        self.register_buffer("band_mask", mask)

    def _mag(self, x):
        s = torch.stft(
            x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop,
            win_length=self.win_len, window=self.window,
            return_complex=True, center=True,
        )
        return s.abs().clamp_min(1e-7)

    def forward(self, pred, target, weight=None):
        p = self._mag(pred)[:, self.band_mask, :]
        t = self._mag(target)[:, self.band_mask, :]

        num = torch.linalg.norm((t - p).flatten(1), dim=1)
        ref = torch.linalg.norm(t.flatten(1), dim=1)

        # A near-silent target drives this denominator to its numerical floor
        # while the numerator stays O(1), producing losses above 1e5. In the
        # first training run this happened in 32 of 120 epochs: gradient
        # clipping prevented divergence but the update direction on those
        # batches was meaningless. Floor the denominator relative to the batch
        # median instead of to a fixed constant, and cap the ratio.
        den = ref.clamp_min(1e-2 * ref.median().clamp_min(1e-4))
        sc = (num / den).clamp_max(10.0)
        mag = (torch.log(p) - torch.log(t)).abs().flatten(1).mean(1)

        per_example = sc + mag

        # Drop examples whose target carries essentially no energy.
        active = (ref > 1e-4).float()
        if active.sum() > 0:
            per_example = per_example * active
        if weight is not None:
            w = weight.to(per_example.device).clamp_min(1e-3)
            return (per_example * w).sum() / w.sum()
        return per_example.mean()


class MultiResBandLoss(nn.Module):
    def __init__(self, sample_rate=4000, band=(150.0, 1400.0),
                 configs=((512, 128, 512), (256, 64, 256), (1024, 256, 1024)),
                 l1_weight=1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.losses = nn.ModuleList([
            BandSTFTLoss(n, h, w, sample_rate, band) for (n, h, w) in configs
        ])

    def forward(self, pred, target, weight=None):
        total = sum(fn(pred, target, weight) for fn in self.losses) / len(self.losses)
        if self.l1_weight:
            l1 = (pred - target).abs().flatten(1).mean(1)
            if weight is not None:
                w = weight.to(l1.device).clamp_min(1e-3)
                l1 = (l1 * w).sum() / w.sum()
            else:
                l1 = l1.mean()
            total = total + self.l1_weight * l1
        return total


def si_sdr(pred, target, eps=1e-8):
    """Reported as a metric, not optimised. Scale-invariant, so it is unaffected
    by the arbitrary gain relationship between the optical and acoustic channels."""
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    alpha = (pred * target).sum(-1, keepdim=True) / (target.pow(2).sum(-1, keepdim=True) + eps)
    proj = alpha * target
    noise = pred - proj
    return 10 * torch.log10(proj.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps) + eps)
