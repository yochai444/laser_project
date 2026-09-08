from pypylon import pylon
import numpy as np
import scipy.io as sio
import serial
import wave
import time

SERIAL_PORT = "COM3"
BAUD_RATE = 921600

EXTERNAL_CLOCK_FREQ = 3000.0
RECORD_DURATION = 20

HEIGHT = 64
WIDTH = 64
EXPOSURE_TIME_US = 125.0

AUDIO_SAMPLE_RATE = 16000
AUDIO_BYTES_PER_SAMPLE = 2
EXPECTED_AUDIO_BYTES = int(AUDIO_SAMPLE_RATE * RECORD_DURATION * AUDIO_BYTES_PER_SAMPLE)

MAX_EXPECTED_FRAMES = int(EXTERNAL_CLOCK_FREQ * RECORD_DURATION)

OUTPUT_VIDEO = r"C:\Users\yocha\OneDrive\Desktop\Y\matrix_112.mat"
OUTPUT_AUDIO = r"C:\Users\yocha\OneDrive\Desktop\Y\microphone_112.wav"


def safe_grab_frame(camera):
    grab_result = camera.RetrieveResult(5, pylon.TimeoutHandling_Return)

    if grab_result is None:
        return None

    if not grab_result.IsValid():
        grab_result.Release()
        return None

    if not grab_result.GrabSucceeded():
        grab_result.Release()
        return None

    frame = grab_result.Array.copy()
    grab_result.Release()

    return frame


def run_external_clock_recording():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        time.sleep(2)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print("Connected to ESP32")
    except Exception as e:
        print("ESP32 not found:", e)
        return

    try:
        tl_factory = pylon.TlFactory.GetInstance()
        camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
        camera.Open()
        print(f"Connected to: {camera.GetDeviceInfo().GetModelName()}")
    except Exception as e:
        print("Camera not found:", e)
        ser.close()
        return

    camera.Width.SetValue(WIDTH)
    camera.Height.SetValue(HEIGHT)
    camera.CenterX.SetValue(True)
    camera.CenterY.SetValue(True)

    camera.ExposureAuto.SetValue("Off")
    camera.GainAuto.SetValue("Off")
    camera.ExposureTime.SetValue(EXPOSURE_TIME_US)

    camera.AcquisitionFrameRateEnable.SetValue(False)

    camera.TriggerSelector.SetValue("FrameStart")
    camera.TriggerMode.SetValue("On")
    camera.TriggerSource.SetValue("Line1")
    camera.TriggerActivation.SetValue("RisingEdge")

    print("\nShort D18 to D19 on ESP32...")

    while True:
        line = ser.readline().decode(errors="ignore").strip()

        if line == "TRIGGERED":
            print("ESP32 trigger received")
            break

    raw_buffer = np.zeros((MAX_EXPECTED_FRAMES, HEIGHT, WIDTH), dtype=np.uint8)
    audio_buffer = bytearray()

    ser.reset_input_buffer()

    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

    time.sleep(0.1)

    print("Sending GO to ESP32")
    ser.write(b"GO\n")
    ser.flush()

    start_time = time.time()
    frame_idx = 0

    while len(audio_buffer) < EXPECTED_AUDIO_BYTES:
        remaining_audio = EXPECTED_AUDIO_BYTES - len(audio_buffer)
        data = ser.read(min(4096, remaining_audio))

        if data:
            audio_buffer.extend(data)

        while True:
            frame = safe_grab_frame(camera)

            if frame is None:
                break

            if frame_idx < MAX_EXPECTED_FRAMES:
                raw_buffer[frame_idx] = frame
                frame_idx += 1

    actual_duration = time.time() - start_time

    camera.StopGrabbing()
    camera.Close()
    ser.close()

    raw_buffer = raw_buffer[:frame_idx]
    measured_fps = frame_idx / RECORD_DURATION

    print("\n--- SUMMARY ---")
    print(f"Requested Duration: {RECORD_DURATION:.2f} sec")
    print(f"Actual Audio Read Duration: {actual_duration:.3f} sec")
    print(f"Expected Audio Bytes: {EXPECTED_AUDIO_BYTES}")
    print(f"Actual Audio Bytes: {len(audio_buffer)}")
    print(f"Audio Duration: {len(audio_buffer) / AUDIO_BYTES_PER_SAMPLE / AUDIO_SAMPLE_RATE:.3f} sec")
    print(f"Frames Captured: {frame_idx}")
    print(f"Measured FPS: {measured_fps:.2f}")

    matlab_ready = np.transpose(raw_buffer, (1, 2, 0))

    sio.savemat(OUTPUT_VIDEO, {
        "vidImages": matlab_ready,
        "fps": measured_fps,
        "record_duration": RECORD_DURATION,
        "frames": frame_idx
    })

    print(f"Saved video: {OUTPUT_VIDEO}")

    with wave.open(OUTPUT_AUDIO, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_SAMPLE_RATE)
        wf.writeframes(audio_buffer)

    print(f"Saved audio: {OUTPUT_AUDIO}")


if __name__ == "__main__":
    run_external_clock_recording()