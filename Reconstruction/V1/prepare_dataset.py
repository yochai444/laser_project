import json
import shutil
from pathlib import Path

def prepare_dataset(json_path, mic_dir, laser_dir, output_dir="dataset"):
    # קריאת קובץ ה-JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"[!] Error: Could not find JSON file at {json_path}")
        return

    # הגדרת נתיבי פלט
    base_out = Path(output_dir)
    train_mic_dir = base_out / "train" / "microphone"
    train_laser_dir = base_out / "train" / "laser"
    test_mic_dir = base_out / "test" / "microphone"
    test_laser_dir = base_out / "test" / "laser"

    # יצירת התיקיות
    for d in [train_mic_dir, train_laser_dir, test_mic_dir, test_laser_dir]:
        d.mkdir(parents=True, exist_ok=True)

    mic_path = Path(mic_dir)
    laser_path = Path(laser_dir)

    train_count = 0
    test_count = 0

    print("[*] Starting dataset sorting...")

    for key, data in metadata.items():
        # יצירת שם הקובץ על בסיס המפתח (למשל "001")
        file_num = str(key).zfill(3)
        mic_file = mic_path / f"microphone_{file_num}.wav"
        laser_file = laser_path / f"Laser_{file_num}.wav"

        # בדיקה שהקבצים אכן קיימים
        if not mic_file.exists() or not laser_file.exists():
            print(f"[!] Warning: Missing files for ID {file_num}. Skipping.")
            continue

        # ניתוח המטא-דאטה: האם נקי או רועש?
        # אנחנו נחשיב כ"נקי" רק את None או מחרוזת ריקה (או רווחים)
        noise_label = str(data.get("Noise", "")).strip().lower()
        
        is_clean = noise_label in ["none", "", "none "]

        # העתקה לתיקייה הרלוונטית
        if is_clean:
            shutil.copy2(mic_file, train_mic_dir / mic_file.name)
            shutil.copy2(laser_file, train_laser_dir / laser_file.name)
            train_count += 1
        else:
            shutil.copy2(mic_file, test_mic_dir / mic_file.name)
            shutil.copy2(laser_file, test_laser_dir / laser_file.name)
            test_count += 1

    print("\n" + "="*40)
    print("✅ Dataset Sorting Completed!")
    print("="*40)
    print(f"📁 Training Set (Clean): {train_count} pairs")
    print(f"📁 Testing Set (Noisy):  {test_count} pairs")
    print("="*40)
    print(f"Data saved to: {base_out.absolute()}")

if __name__ == "__main__":
    # >>> שים לב: עדכן את שם קובץ ה-JSON כאן למטה לשם האמיתי שיש לך <<<
    json_filename = "Recordings Metadata.json" 
    
    # מניח שתיקיות המקור הן laser ו-microphone באותה תיקייה
    prepare_dataset(
        json_path=json_filename,
        mic_dir="microphone",
        laser_dir="laser",
        output_dir="dataset"
    )