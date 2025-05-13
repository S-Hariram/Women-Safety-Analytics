import cv2
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow import keras
import winsound  # For Windows alert sound

# Suppress TensorFlow warnings
tf.get_logger().setLevel("ERROR")

# === Paths and Config ===
MODEL_PATH = r"C:\Real-Time-Violence-Detection-main\Violence Detection\modelnew.h5"
VIOLENCE_FRAMES_DIR = "violence_frames"
os.makedirs(VIOLENCE_FRAMES_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
VIOLENCE_THRESHOLD = 0.4
MOTION_THRESHOLD = 1.2
VIOLENCE_CONFIRM_COUNT = 2

# === Load model ===
if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found. Check the path.")
    exit()

try:
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# === Webcam setup ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

prev_gray = None
last_alert_time = 0
ALERT_INTERVAL = 2  # seconds
violence_counter = 0
alert_active = False
violenceCount = [0, int(time.time())]
danger_shown = False
danger_start_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture failed.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion_magnitude = 0

    # === Motion Detection (Optical Flow) ===
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        motion_magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))

    prev_gray = gray.copy()

    # === Violence Prediction ===
    resized = cv2.resize(frame, IMG_SIZE)
    input_data = np.expand_dims(resized / 255.0, axis=0)
    prediction = model.predict(input_data, verbose=0)[0][0]

    current_time = time.time()
    violence_detected = prediction > VIOLENCE_THRESHOLD and motion_magnitude > MOTION_THRESHOLD

    if violence_detected:
        violence_counter += 1
        print(f"[DEBUG] Violence count: {violence_counter} | Model: {prediction:.2f} | Motion: {motion_magnitude:.2f}")
    else:
        violence_counter = 0
        alert_active = False
        print(f"[DEBUG] Model: {prediction:.2f} | Motion: {motion_magnitude:.2f}")

    # === Trigger Alert ===
    if violence_counter >= VIOLENCE_CONFIRM_COUNT and not alert_active:
        if current_time - last_alert_time > ALERT_INTERVAL:
            
            violenceCount[0] += 1
            print(violenceCount)

            # === Danger condition ===
            if violenceCount[0] > 2 and (int(current_time) - violenceCount[1]) <= 10:
                print("🚨 danger")
                danger_shown = True
                danger_start_time = current_time

            if (int(current_time) - violenceCount[1]) > 10:
                violenceCount[0] = 0
                violenceCount[1] = int(current_time)

            # Save Frame
            filename = f"violence_{int(current_time)}.jpg"
            filepath = os.path.join(VIOLENCE_FRAMES_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"⚠️ Violence Detected! Frame saved: {filepath}")

            try:
                winsound.Beep(1000, 500)
            except:
                pass

            last_alert_time = current_time
            alert_active = True

    # === Display ===
    label = "⚠️ VIOLENCE DETECTED" if violence_counter >= VIOLENCE_CONFIRM_COUNT else "✅ NO VIOLENCE"
    color = (0, 0, 255) if violence_counter >= VIOLENCE_CONFIRM_COUNT else (0, 255, 0)

    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Model Score: {prediction:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"Motion: {motion_magnitude:.2f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # === DANGER ALERT ON SCREEN ===
    if danger_shown and current_time - danger_start_time <= 5:
        cv2.putText(frame, "🚨 DANGER 🚨", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    elif current_time - danger_start_time > 5:
        danger_shown = False

    cv2.imshow("Violence Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
