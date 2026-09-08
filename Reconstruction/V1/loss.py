import torch
import torch.nn as nn
import torch.nn.functional as F

class STFTLoss(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))

    def forward(self, pred, target):
        pred_stft = torch.stft(pred.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, 
                               win_length=self.win_length, window=self.window, return_complex=True)
        target_stft = torch.stft(target.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, 
                                 win_length=self.win_length, window=self.window, return_complex=True)
        
        pred_mag = torch.abs(pred_stft) + 1e-7
        target_mag = torch.abs(target_stft) + 1e-7
        
        sc_loss = torch.norm(target_mag - pred_mag, p="fro") / torch.norm(target_mag, p="fro")
        mag_loss = F.l1_loss(torch.log(pred_mag), torch.log(target_mag))
        
        return sc_loss + mag_loss