import cv2
import numpy as np
import os
import time
import datetime
import threading
import smtplib
import winsound
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from geopy.geocoders import Nominatim
from tensorflow import keras
from collections import deque

# ------------------------ CONFIGURATION ------------------------ #
HARASSMENT_MODEL_PATH = r"C:\sheild-master\model\keras_model.h5"
HARASSMENT_LABELS_PATH = r"C:\sheild-master\model\labels.txt"
VIOLENCE_MODEL_PATH = r"C:\Real-Time-Violence-Detection-main\Violence Detection\modelnew.h5"
VIDEO_PATH = r"C:\sheild-master\Real-Time-Violence-Detection-main\Violence Detection\Testing videos\V_19.mp4"

EMAIL_ADDRESS = "hariramsundaram@gmail.com"
EMAIL_PASSWORD = "jywr idqd spma mumu"  # Use App Password
TO_EMAIL = "hariram12003@gmail.com"

DETECTION_FRAMES_DIR = "detection_frames"
os.makedirs(DETECTION_FRAMES_DIR, exist_ok=True)

MIN_REQUIRED_CONSECUTIVE_DETECTIONS = 8
VIOLENCE_THRESHOLD = 0.4
MOTION_THRESHOLD = 4.5
HARASSMENT_CONFIDENCE_THRESHOLD = 0.85
# -------------------------------------------------------------- #

# Load models
try:
    harassment_model = keras.models.load_model(HARASSMENT_MODEL_PATH, compile=False)
    harassment_model.compile(optimizer=keras.optimizers.Adam(0.001),
                              loss='categorical_crossentropy', metrics=['accuracy'])
    violence_model = keras.models.load_model(VIOLENCE_MODEL_PATH)
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    exit()

# Load labels
with open(HARASSMENT_LABELS_PATH, "r") as f:
    harassment_labels = [line.strip() for line in f.readlines()]

# Email alert
def send_email(image_path, current_time, location_details):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_EMAIL
        msg['Subject'] = "⚠️ ALERT: Harassment/Violence Detected"
        body = f"🚨 Alert!\nTime: {current_time}\nLocation: {location_details}"
        msg.attach(MIMEText(body, 'plain'))

        if os.path.exists(image_path):
            with open(image_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(image_path)}")
                msg.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, TO_EMAIL, msg.as_string())
        server.quit()
        print("📧 Email alert sent.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# Location fetcher
def get_location():
    try:
        geolocator = Nominatim(user_agent="SHEild/1.0")
        location = geolocator.geocode("12A State Highway, CGC Jhanjeri, Mohali, Punjab, India")
        return f"{location.address}, Lat: {location.latitude}, Lon: {location.longitude}" if location else "Location not found"
    except Exception as e:
        return f"Location error: {e}"

# Frame preprocessing
def preprocess_frame(frame):
    resized = cv2.resize(frame, (224, 224))
    norm = resized.astype(np.float32) / 255.0
    return np.expand_dims(norm, axis=0)

# Video capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Failed to open video.")
    exit()

frame_count = 0
consecutive_positive_frames = 0
snapshot_taken = False
violence_buffer = []
prev_gray = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 End of video.")
        break

    frame_count += 1
    if frame_count % 1 != 0:
        continue  # Skip 2 of every 3 frames for faster inference

    # HARASSMENT DETECTION
    h_input = preprocess_frame(frame)
    h_pred = harassment_model.predict(h_input, verbose=0)
    h_index = np.argmax(h_pred)
    h_label = harassment_labels[h_index]
    h_score = h_pred[0][h_index]

    if h_label == "0-Positive" and h_score > HARASSMENT_CONFIDENCE_THRESHOLD:
        consecutive_positive_frames += 1
    else:
        consecutive_positive_frames = 0

    harassment_detected = consecutive_positive_frames >= MIN_REQUIRED_CONSECUTIVE_DETECTIONS

    print(f"[HARASSMENT] Class: {h_label} | Confidence: {round(h_score*100)}% | Streak: {consecutive_positive_frames}")

    if harassment_detected and not snapshot_taken:
        snapshot_path = os.path.join(DETECTION_FRAMES_DIR, "harassment_snapshot.jpg")
        cv2.imwrite(snapshot_path, frame)
        print("📸 Harassment snapshot saved.")
        snapshot_taken = True

    # VIOLENCE DETECTION
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion_detected = False
    motion_magnitude = 0

    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        motion_magnitude = np.mean(np.abs(flow))
        motion_detected = motion_magnitude > MOTION_THRESHOLD
    prev_gray = gray.copy()

    violence_input = cv2.resize(frame, (128, 128)) / 255.0
    v_pred = violence_model.predict(np.expand_dims(violence_input, axis=0), verbose=0)
    v_score = v_pred[0][0]

    violence_buffer.append(v_score if motion_detected else 0)
    if len(violence_buffer) > 15:
        violence_buffer.pop(0)
    v_avg = np.mean(violence_buffer)

    violence_detected = v_avg > VIOLENCE_THRESHOLD

    print(f"[VIOLENCE] Conf: {round(v_avg * 100)}% | Motion: {motion_magnitude:.2f}")

    # ALERT
    if harassment_detected or violence_detected:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        location = get_location()
        alert_path = os.path.join(DETECTION_FRAMES_DIR, f"alert_{int(time.time())}.jpg")
        cv2.imwrite(alert_path, frame)

        try:
            winsound.Beep(1000, 600)
        except:
            pass

        threading.Thread(target=send_email, args=(alert_path, timestamp, location)).start()
        print("🚨 ALERT TRIGGERED.")
        break

    # DISPLAY
    if harassment_detected:
        label = "⚠️ HARASSMENT DETECTED"
        color = (0, 0, 255)
    elif violence_detected:
        label = "⚠️ VIOLENCE DETECTED"
        color = (0, 0, 255)
    else:
        label = "✅ SAFE ENVIRONMENT"
        color = (0, 255, 0)

    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    cv2.imshow("SHEild - Safety Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 User stopped.")
        break

cap.release()
cv2.destroyAllWindows()
