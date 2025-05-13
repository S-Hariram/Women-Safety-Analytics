import cv2
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow import keras
import winsound  # Windows-only for alert sounds
from collections import deque

# Suppress TensorFlow warnings
tf.get_logger().setLevel("ERROR")

# Model Paths
HARASSMENT_MODEL_PATH = r"C:\sheild-master\model\keras_model.h5"
HARASSMENT_LABELS_PATH = r"C:\sheild-master\model\labels.txt"
VIOLENCE_MODEL_PATH = r"C:\Real-Time-Violence-Detection-main\Violence Detection\modelnew.h5"

# Load Models
try:
    harassment_model = keras.models.load_model(HARASSMENT_MODEL_PATH, compile=False)
    harassment_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                             loss='categorical_crossentropy', metrics=['accuracy'])
    violence_model = keras.models.load_model(VIOLENCE_MODEL_PATH)
    print("✅ Models Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# Load Labels for Harassment Model
with open(HARASSMENT_LABELS_PATH, "r") as f:
    harassment_labels = [line.strip() for line in f.readlines()]

# Model Input Size
HARASSMENT_IMG_SIZE = (224, 224)
VIOLENCE_IMG_SIZE = (128, 128)

# Use Video Instead of Webcam
VIDEO_PATH = r"C:\sheild-master\Real-Time-Violence-Detection-main\Violence Detection\Testing videos\V_19.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Error: Could not open video file.")
    exit()

# Detection Parameters
VIOLENCE_THRESHOLD = 0.4  # Adjusted based on data
HARASSMENT_THRESHOLD = 0.7
DETECTION_COUNT = 6
MOTION_THRESHOLD = 4.5  # Reduced to make detection more sensitive

# Motion Detection Variables
prev_gray = None
motion_detected = False  # Ensure motion_detected is always defined
harassment_buffer = deque(maxlen=15)
violence_buffer = deque(maxlen=15)

# Directory for Saving Frames
DETECTION_FRAMES_DIR = "detection_frames"
os.makedirs(DETECTION_FRAMES_DIR, exist_ok=True)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("✅ Video processing completed.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion_magnitude = 0

    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion_magnitude = np.mean(np.abs(flow))
        motion_detected = motion_magnitude > MOTION_THRESHOLD
    prev_gray = gray.copy()

    # Preprocess Frames for Models
    harassment_img = cv2.resize(frame, HARASSMENT_IMG_SIZE) / 255.0
    violence_img = cv2.resize(frame, VIOLENCE_IMG_SIZE) / 255.0

    harassment_pred = harassment_model.predict(np.expand_dims(harassment_img, axis=0), verbose=0)
    violence_pred = violence_model.predict(np.expand_dims(violence_img, axis=0), verbose=0)

    harassment_index = np.argmax(harassment_pred)
    harassment_label = harassment_labels[harassment_index]
    harassment_score = harassment_pred[0][harassment_index]
    violence_score = violence_pred[0][0]

    harassment_buffer.append(harassment_score if harassment_label == "0-Positive" else 0)
    violence_buffer.append(violence_score if motion_detected else 0)

    harassment_avg = np.mean(list(harassment_buffer)) if harassment_buffer else 0
    violence_avg = np.mean(list(violence_buffer)) if violence_buffer else 0

    print(f"Harassment: {round(harassment_avg * 100)}% | Violence: {round(violence_avg * 100)}% | Motion: {motion_magnitude:.2f}")

    harassment_detected = harassment_avg > HARASSMENT_THRESHOLD
    violence_detected = violence_avg > VIOLENCE_THRESHOLD

    if harassment_detected:
        label = "⚠️ HARASSMENT DETECTED"
        color = (0, 0, 255)
    elif violence_detected:
        label = "⚠️ VIOLENCE DETECTED"
        color = (0, 0, 255)
    else:
        label = "✅ SAFE ENVIRONMENT"
        color = (0, 255, 0)

    if harassment_detected or violence_detected:
        frame_name = f"detection_{int(time.time())}.jpg"
        frame_path = os.path.join(DETECTION_FRAMES_DIR, frame_name)
        cv2.imwrite(frame_path, frame)
        print(f"🔴 Incident Confirmed! Frame saved: {frame_path}")

        try:
            winsound.Beep(1000, 500)
        except:
            pass

    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Live Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Stopping video processing...")
        break

cap.release()
cv2.destroyAllWindows()
