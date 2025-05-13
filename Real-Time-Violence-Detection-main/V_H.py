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

# Webcam Initialization
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Detection Thresholds (Adjusted for Better Accuracy)
VIOLENCE_THRESHOLD = 0.5   # Increased to avoid false positives
HARASSMENT_THRESHOLD = 0.7  # Adjusted for best accuracy
DETECTION_COUNT = 12        # Frames required for confirmation (was 10)
ALERT_THRESHOLD = 15        # Continuous frames before alert

# Motion Detection Parameters
BUFFER_LENGTH = 15
harassment_buffer = deque(maxlen=BUFFER_LENGTH)
violence_buffer = deque(maxlen=BUFFER_LENGTH)

# Create Directory for Saving Frames
DETECTION_FRAMES_DIR = "detection_frames"
os.makedirs(DETECTION_FRAMES_DIR, exist_ok=True)

# Motion Detection Variables
prev_gray = None
motion_magnitude = 0  # Initialize to prevent NameError

incident_confirmed = False  # Webcam remains blank until detection

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame.")
        break

    # Convert frame to grayscale for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Motion Using Optical Flow (Lucas-Kanade)
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion_magnitude = np.mean(np.abs(flow))
        motion_detected = motion_magnitude > 4  # Increased sensitivity

    prev_gray = gray.copy()

    # Preprocess frame for both models
    harassment_img = cv2.resize(frame, HARASSMENT_IMG_SIZE)
    harassment_img = np.expand_dims(harassment_img / 255.0, axis=0)

    violence_img = cv2.resize(frame, VIOLENCE_IMG_SIZE)
    violence_img = np.expand_dims(violence_img / 255.0, axis=0)

    # Predictions
    harassment_pred = harassment_model.predict(harassment_img, verbose=0)
    violence_pred = violence_model.predict(violence_img, verbose=0)

    # Extract Results
    harassment_index = np.argmax(harassment_pred)
    harassment_label = harassment_labels[harassment_index]
    harassment_score = harassment_pred[0][harassment_index]

    violence_score = violence_pred[0][0]  # Single output model (higher = more violent)

    # Append to Moving Average Buffers
    harassment_buffer.append(harassment_score if harassment_label == "0-Positive" else 0)
    violence_buffer.append(violence_score if motion_magnitude > 3 else 0)  # Only add if motion is strong

    # Weighted Average to Reduce Fluctuations
    if harassment_buffer and violence_buffer:
        weights = np.linspace(1, 2, len(harassment_buffer))
        harassment_avg = np.average(harassment_buffer, weights=weights)
        violence_avg = np.average(violence_buffer, weights=weights)
    else:
        harassment_avg = 0
        violence_avg = 0

    # Print Confidence Levels
    print(f"Harassment: {round(harassment_avg * 100)}% | Violence: {round(violence_avg * 100)}% | Motion: {motion_magnitude:.2f}")

    # Check for Continuous Incident
    harassment_detected = sum(1 for val in list(harassment_buffer)[-DETECTION_COUNT:] if val > HARASSMENT_THRESHOLD) >= DETECTION_COUNT
    violence_detected = sum(1 for val in list(violence_buffer)[-DETECTION_COUNT:] if val > VIOLENCE_THRESHOLD) >= DETECTION_COUNT

    if harassment_detected or violence_detected:
        if not incident_confirmed:
            print("🔴 Incident Confirmed! Showing Webcam Feed...")
            incident_confirmed = True

        label = "⚠️ HARASSMENT DETECTED" if harassment_detected else "⚠️ VIOLENCE DETECTED"
        color = (0, 0, 255)  # Red for Alert

        # Save Frame on Detection
        frame_name = f"detection_{int(time.time())}.jpg"
        frame_path = os.path.join(DETECTION_FRAMES_DIR, frame_name)
        cv2.imwrite(frame_path, frame)
        print(f"🔴 Incident Confirmed! Frame saved: {frame_path}")

        # Play Alert Sound (Windows Only)
        try:
            winsound.Beep(1000, 500)
        except:
            pass  # Ignore errors on non-Windows systems

    else:
        label = "✅ SAFE ENVIRONMENT"
        color = (0, 255, 0)  # Green for No Threat

    # Only Show Webcam if an Incident is Confirmed
    # Always Show Webcam
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Live Monitoring", frame)
    incident_confirmed = True  # Ensure it's always true

    
    # Exit on 'q' Key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Stopping video processing...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
