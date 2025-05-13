import cv2
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow import keras
import winsound
from collections import deque

# === Suppress TensorFlow Warnings ===
tf.get_logger().setLevel("ERROR")

# === Paths and Config ===
MODEL_PATH = r"C:\Real-Time-Violence-Detection-main\Violence Detection\modelnew.h5"
VIOLENCE_FRAMES_DIR = "violence_frames"
os.makedirs(VIOLENCE_FRAMES_DIR, exist_ok=True)

# Change IMG_SIZE to match the model's expected input shape
IMG_SIZE = (128, 128)  # Changed from (128, 128) to (224, 224)
VIOLENCE_SOFT_THRESHOLD = 0.55  # Raised for better balance
VIOLENCE_HARD_THRESHOLD = 0.85
MOTION_THRESHOLD = 1.2

PREDICTION_WINDOW = 7
MOTION_WINDOW = 7
ALERT_INTERVAL = 1  # seconds
ESCALATION_TIME = 5  # seconds

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

# === Webcam Setup ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

prev_gray = None
last_alert_time = 0
prediction_queue = deque(maxlen=PREDICTION_WINDOW)
motion_queue = deque(maxlen=MOTION_WINDOW)
escalation_start = None

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture failed.")
        break

    # === Motion Detection ===
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion_magnitude = 0
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        motion_magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
    prev_gray = gray.copy()

    # === Violence Prediction ===
    resized = cv2.resize(frame, IMG_SIZE)  # Resize the frame to (224, 224)
    input_data = np.expand_dims(resized / 255.0, axis=0)
    prediction = model.predict(input_data, verbose=0)[0][0]

    # Add scores to queues
    prediction_queue.append(prediction)
    motion_queue.append(motion_magnitude)

    avg_prediction = np.mean(prediction_queue)
    avg_motion = np.mean(motion_queue)

    # === Detection Logic ===
    violence_detected = False

    if avg_prediction > VIOLENCE_HARD_THRESHOLD:
        violence_detected = True
    elif avg_prediction > 0.35 and avg_motion > (MOTION_THRESHOLD + 1.0):
        violence_detected = True

    # === Alert Logic ===
    label = "✅ NO VIOLENCE"
    color = (0, 255, 0)

    if violence_detected:
        label = "⚠️ VIOLENCE DETECTED"
        color = (0, 0, 255)

        current_time = time.time()
        if escalation_start is None:
            escalation_start = current_time

        if current_time - last_alert_time > ALERT_INTERVAL:
            filename = f"violence_{int(current_time)}.jpg"
            filepath = os.path.join(VIOLENCE_FRAMES_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"⚠️ Violence Detected! Frame saved: {filepath}")
            print(f"[DEBUG] Model Avg: {avg_prediction:.2f} | Motion Avg: {avg_motion:.2f}")

            try:
                winsound.Beep(1000, 500)
            except:
                print("🔇 Beep not supported on this device.")

            last_alert_time = current_time

        # Escalation
        if current_time - escalation_start > ESCALATION_TIME:
            print("🚨 Extended Violence Detected! Escalating response.")
            try:
                winsound.Beep(1500, 700)
            except:
                pass
    else:
        escalation_start = None
        print(f"[DEBUG] Model Avg: {avg_prediction:.2f} | Motion Avg: {avg_motion:.2f}")

    # === Logging ===
    with open("violence_log.txt", "a") as log:
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, "
                  f"Prediction: {avg_prediction:.2f}, "
                  f"Motion: {avg_motion:.2f}, "
                  f"Detected: {violence_detected}\n")

    # === Overlay Info on Frame ===
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Model Score (Avg): {avg_prediction:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"Motion (Avg): {avg_motion:.2f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow("Violence Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
print("🔚 Exiting program safely. Resources released.")
