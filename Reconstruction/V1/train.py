import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import OpticalMicDataset
from model import SimpleAudioUNet
from loss import STFTLoss

def main():
    # פנייה לתיקיית האימון החדשה שיצרת (המכילה רק את ההקלטות השקטות)
    base_dir = Path("dataset/train") 
    laser_dir = base_dir / "laser"
    mic_dir = base_dir / "microphone"

    laser_files = sorted(list(laser_dir.glob("Laser_*.wav")))
    mic_files = sorted(list(mic_dir.glob("microphone_*.wav")))

    assert len(laser_files) == len(mic_files), "Mismatch in file counts!"
    total_files = len(laser_files)
    print(f"Found {total_files} clean audio pairs for training.")

    # >>> שילוב קובץ הרעש האמיתי שלך כאן <<<
    noise_file_path = r"C:\Users\yocha\OneDrive\Desktop\cleanUnet\microphone\microphone_177.wav"
    
    dataset = OpticalMicDataset(
        mic_paths=mic_files, 
        opt_paths=laser_files,
        noise_path=noise_file_path
    )
    
    val_size = int(0.2 * total_files)
    train_size = total_files - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Split: {train_size} for Training, {val_size} for Validation.")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleAudioUNet(in_channels=2).to(device)
    
    l1_criterion = nn.L1Loss()
    stft_criterion = STFTLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50
    best_val_loss = float('inf') 
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        
        # מקבלים 3 משתנים מה-Dataset
        for batch_idx, (noisy_mic, opt_chunk, clean_mic) in enumerate(train_loader):
            noisy_mic = noisy_mic.to(device)
            opt_chunk = opt_chunk.to(device)
            clean_mic = clean_mic.to(device)
            
            # שרשור המיקרופון (שהרועש מלאכותית מהרעש האמיתי) והאות האופטי לממד קלט אחד
            combined_input = torch.cat((noisy_mic, opt_chunk), dim=1)
            
            optimizer.zero_grad()
            predictions = model(combined_input)
            
            # חישוב ה-Loss מול המיקרופון הנקי המקורי
            loss = l1_criterion(predictions, clean_mic) + stft_criterion(predictions, clean_mic)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for noisy_mic, opt_chunk, clean_mic in val_loader:
                noisy_mic = noisy_mic.to(device)
                opt_chunk = opt_chunk.to(device)
                clean_mic = clean_mic.to(device)
                
                combined_input = torch.cat((noisy_mic, opt_chunk), dim=1)
                predictions = model(combined_input)
                loss = l1_criterion(predictions, clean_mic) + stft_criterion(predictions, clean_mic)
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "unet_model_best.pt")
            print(f"  -> New best model saved! (Validation loss dropped to {best_val_loss:.4f})")

if __name__ == "__main__":
    main()