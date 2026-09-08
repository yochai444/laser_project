"""
freq_warp.py
============
Warps the frequency axis of a spectrogram, so the model sees each recording as
if it came from a voice with a different pitch and vocal tract length.

WHY THIS AND NOT WAVEFORM PITCH SHIFTING
    Shifting pitch in the waveform without changing duration needs a phase
    vocoder. A naive overlap-add stretch destroys periodicity - tested here, a
    -4 semitone shift of a 150 Hz tone came out at 221 Hz instead of 119 Hz.
    Warping the spectrogram achieves the same augmentation with no artefacts and
    no extra dependency, and it acts directly on what the network consumes.

    This is vocal tract length perturbation, a standard augmentation. It moves
    the fundamental and the formants together, which is what happens between
    speakers anyway.

WHY IT SHOULD HELP HERE SPECIFICALLY
    Measured microphone-to-laser coherence peaks at 188-312 Hz and collapses
    past 700 Hz, so the fundamental is the strongest surviving feature in the
    whole signal - and the model has five voices to learn it from. The existing
    augmentation perturbs speed by +/- 8%, which on a 150 Hz fundamental spans
    139-163 Hz. A warp of +/- 15% spans 128-173 Hz, and unlike speed
    perturbation it does not also change the duration of the word.
"""

import numpy as np
import torch


def warp_linear_bins(x, factor, bin0):
    """Warp a spectrogram whose rows are consecutive linear FFT bins.

    x       [.., F, T] tensor, row i is FFT bin (bin0 + i)
    factor  >1 shifts content up in frequency, <1 down
    bin0    absolute index of the first row, needed because frequency is
            proportional to the ABSOLUTE bin number. Warping row indices
            directly would be wrong whenever the band does not start at 0 Hz -
            here it starts at bin 10, so ignoring the offset would warp
            156-688 Hz as though it were 0-530 Hz.
    """
    if abs(factor - 1.0) < 1e-6:
        return x
    n = x.shape[-2]
    i = torch.arange(n, dtype=torch.float32)
    src = (bin0 + i) / factor - bin0          # where to read row i from
    src = src.clamp(0, n - 1)

    lo = src.floor().long()
    hi = (lo + 1).clamp(max=n - 1)
    w = (src - lo.float()).view(-1, 1)

    return x[..., lo, :] * (1 - w) + x[..., hi, :] * w


def warp_bands(x, factor):
    """Warp a spectrogram whose rows are mel bands.

    Bands are already spaced roughly logarithmically, so a frequency scaling is
    close to a constant shift in band index. Interpolating on the index is a
    good enough approximation and avoids carrying the band edges around.
    """
    if abs(factor - 1.0) < 1e-6:
        return x
    n = x.shape[-2]
    shift = np.log(factor) / np.log(2 ** (1 / 12))   # in semitone-ish units
    i = torch.arange(n, dtype=torch.float32)
    src = (i - shift * 0.5).clamp(0, n - 1)

    lo = src.floor().long()
    hi = (lo + 1).clamp(max=n - 1)
    w = (src - lo.float()).view(-1, 1)
    return x[..., lo, :] * (1 - w) + x[..., hi, :] * w


def random_warp(x, feat, max_pct=15.0, rng=None):
    """Apply a random warp of up to +/- max_pct percent."""
    r = np.random.default_rng() if rng is None else rng
    factor = 1.0 + r.uniform(-max_pct, max_pct) / 100.0
    if getattr(feat, "use_filterbank", False):
        return warp_bands(x, factor)
    bin0 = int(feat.bins[0]) if getattr(feat, "bins", None) is not None else 0
    return warp_linear_bins(x, factor, bin0)
