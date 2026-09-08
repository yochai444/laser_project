import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset
import random
from pathlib import Path

class OpticalMicDataset(Dataset):
    # כאן מוגדר הארגומנט שחסר לפייתון: noise_path=None
    def __init__(self, mic_paths, opt_paths, noise_path=None, chunk_length_sec=2.0, sample_rate=16000, 
                 max_delay_sec=0.1, optical_nyquist_hz=1500.0):
        self.mic_paths = mic_paths
        self.opt_paths = opt_paths
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_length_sec * sample_rate)
        self.max_delay_samples = int(max_delay_sec * sample_rate)
        self.lpf_freq = optical_nyquist_hz
        
        # טעינת קובץ הרעש האמיתי לזיכרון
        self.noise_waveform = None
        if noise_path and Path(noise_path).exists():
            print(f"[*] Loaded real noise file: {noise_path}")
            noise_wave, sr = torchaudio.load(noise_path)
            if noise_wave.shape[0] > 1:
                noise_wave = noise_wave[0:1, :]
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                noise_wave = resampler(noise_wave)
            self.noise_waveform = noise_wave
        else:
            print("[!] No valid real noise file found. Will fall back to synthetic noise.")

    def __len__(self):
        return len(self.mic_paths)

    def _align_signals(self, mic, opt):
        mic_filtered = F.lowpass_biquad(mic, self.sample_rate, self.lpf_freq)
        n = mic.shape[-1]
        n_fft = 2 ** (2 * n - 1).bit_length()
        
        mic_fft = torch.fft.rfft(mic_filtered, n=n_fft)
        opt_fft = torch.fft.rfft(opt, n=n_fft)
        
        cc = torch.fft.irfft(mic_fft * torch.conj(opt_fft), n=n_fft)
        cc = torch.roll(cc, shifts=n, dims=-1)
        
        center = n
        search_window = cc[0, center - self.max_delay_samples : center + self.max_delay_samples]
        peak_idx = torch.argmax(search_window)
        delay = peak_idx.item() - self.max_delay_samples
        
        if delay > 0:
            mic_aligned = mic[:, delay:]
            opt_aligned = opt[:, :-delay] if delay > 0 else opt
        elif delay < 0:
            mic_aligned = mic[:, :delay] if delay < 0 else mic
            opt_aligned = opt[:, -delay:]
        else:
            mic_aligned, opt_aligned = mic, opt
            
        min_len = min(mic_aligned.shape[-1], opt_aligned.shape[-1])
        return mic_aligned[:, :min_len], opt_aligned[:, :min_len]

    def _add_real_noise(self, clean_mic):
        snr_db = random.uniform(-5, 10)
        
        if self.noise_waveform is None:
            noise = torch.randn_like(clean_mic)
        else:
            noise_len = self.noise_waveform.shape[-1]
            if noise_len > self.chunk_size:
                start_idx = random.randint(0, noise_len - self.chunk_size)
                noise = self.noise_waveform[:, start_idx : start_idx + self.chunk_size]
            else:
                noise = torch.nn.functional.pad(self.noise_waveform, (0, self.chunk_size - noise_len))
                
        snr_linear = 10 ** (snr_db / 20)
        
        clean_rms = torch.sqrt(torch.mean(clean_mic**2) + 1e-8)
        noise_rms = torch.sqrt(torch.mean(noise**2) + 1e-8)
        
        adjusted_noise = noise * (clean_rms / noise_rms) / snr_linear
        noisy_mic = clean_mic + adjusted_noise
        return noisy_mic

    def __getitem__(self, idx):
        clean_mic_waveform, _ = torchaudio.load(self.mic_paths[idx])
        opt_waveform, _ = torchaudio.load(self.opt_paths[idx])
        
        mic_aligned, opt_aligned = self._align_signals(clean_mic_waveform, opt_waveform)
        
        max_start = mic_aligned.shape[-1] - self.chunk_size
        if max_start > 0:
            start_idx = random.randint(0, max_start)
            clean_mic_chunk = mic_aligned[:, start_idx : start_idx + self.chunk_size]
            opt_chunk = opt_aligned[:, start_idx : start_idx + self.chunk_size]
        else:
            clean_mic_chunk = torch.nn.functional.pad(mic_aligned, (0, self.chunk_size - mic_aligned.shape[-1]))
            opt_chunk = torch.nn.functional.pad(opt_aligned, (0, self.chunk_size - opt_aligned.shape[-1]))

        noisy_mic_chunk = self._add_real_noise(clean_mic_chunk)
        
        clean_mic_chunk = clean_mic_chunk / (torch.max(torch.abs(clean_mic_chunk)) + 1e-8)
        noisy_mic_chunk = noisy_mic_chunk / (torch.max(torch.abs(noisy_mic_chunk)) + 1e-8)
        opt_chunk = opt_chunk / (torch.max(torch.abs(opt_chunk)) + 1e-8)

        return noisy_mic_chunk, opt_chunk, clean_mic_chunk