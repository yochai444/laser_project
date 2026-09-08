import torch
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from model import SimpleAudioUNet
from dataset import OpticalMicDataset

def plot_spectrograms(noisy_waveform, enhanced_waveform, sample_rate, save_path):
    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256)
    db_transform = torchaudio.transforms.AmplitudeToDB()

    noisy_spec = db_transform(spectrogram_transform(noisy_waveform))
    enhanced_spec = db_transform(spectrogram_transform(enhanced_waveform))

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    im1 = axs[0].imshow(noisy_spec[0].numpy(), origin='lower', aspect='auto', cmap='magma')
    axs[0].set_title("Original (Noisy Microphone)")
    axs[0].set_ylabel("Frequency Bins")
    fig.colorbar(im1, ax=axs[0], format="%+2.0f dB")

    im2 = axs[1].imshow(enhanced_spec[0].numpy(), origin='lower', aspect='auto', cmap='magma')
    axs[1].set_title("Enhanced (Conditioned CleanUNet Output)")
    axs[1].set_ylabel("Frequency Bins")
    axs[1].set_xlabel("Time Frames")
    fig.colorbar(im2, ax=axs[1], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def enhance_audio(model_path, input_mic, input_opt, output_wav, output_plot, sample_rate=16000):
    device = torch.device("cpu")
    
    if not Path(model_path).exists():
        print(f"\n[Error] Model file not found: {model_path}")
        return
        
    print(f"[*] Loading conditioned model: {model_path}")
    model = SimpleAudioUNet(in_channels=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"[*] Loading audio files...")
    mic_wave, sr = torchaudio.load(input_mic)
    opt_wave, _ = torchaudio.load(input_opt)
    
    if mic_wave.shape[0] > 1: mic_wave = mic_wave[0:1, :]
    if opt_wave.shape[0] > 1: opt_wave = opt_wave[0:1, :]

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        mic_wave = resampler(mic_wave)
        opt_wave = resampler(opt_wave)

    aligner = OpticalMicDataset(mic_paths=[], opt_paths=[]) 
    mic_aligned, opt_aligned = aligner._align_signals(mic_wave, opt_wave)

    mic_norm = mic_aligned / (torch.max(torch.abs(mic_aligned)) + 1e-8)
    opt_norm = opt_aligned / (torch.max(torch.abs(opt_aligned)) + 1e-8)

    combined_input = torch.cat((mic_norm, opt_norm), dim=0).unsqueeze(0).to(device)
    
    print("[*] Processing audio... This might take a few seconds.")
    with torch.no_grad():
        enhanced_tensor = model(combined_input)
    
    enhanced_norm = enhanced_tensor.squeeze(0).cpu()

    torchaudio.save(output_wav, enhanced_norm, sample_rate)
    print(f"[*] Enhanced audio saved to: {output_wav}")

    print("[*] Generating visual spectrogram comparison...")
    plot_spectrograms(mic_norm, enhanced_norm, sample_rate, output_plot)
    print(f"[*] Visual plot saved to: {output_plot}")
    print("[+] Done! You can now open the files.")

def main():
    root = tk.Tk()
    root.withdraw()

    print("Please select the noisy microphone .wav file in the window...")
    input_mic = filedialog.askopenfilename(title="1. בחר את קובץ המיקרופון (Microphone)", filetypes=[("WAV Audio Files", "*.wav")])
    if not input_mic: return

    print("Please select the matching laser .wav file in the window...")
    input_opt = filedialog.askopenfilename(title="2. בחר את קובץ הלייזר התואם (Laser)", filetypes=[("WAV Audio Files", "*.wav")])
    if not input_opt: return

    print("Please select where to save the enhanced file...")
    output_wav = filedialog.asksaveasfilename(title="3. שמור את הקובץ הנקי כ...", defaultextension=".wav", filetypes=[("WAV Audio Files", "*.wav")])
    if not output_wav: return

    output_plot = str(Path(output_wav).with_suffix('.png'))
    model_path = "unet_model_best.pt"

    enhance_audio(model_path, input_mic, input_opt, output_wav, output_plot)

if __name__ == "__main__":
    main()