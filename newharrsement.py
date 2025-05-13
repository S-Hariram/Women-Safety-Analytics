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
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout

# ------------------------ CONFIGURATION ------------------------ #
HARASSMENT_MODEL_PATH = r"C:\sheild-master\model\keras_model.h5"
HARASSMENT_LABELS_PATH = r"C:\sheild-master\model\labels.txt"
VIOLENCE_MODEL_PATH = r"C:\Real-Time-Violence-Detection-main\Violence Detection\modelnew.h5"

EMAIL_ADDRESS = "hariramsundaram@gmail.com"
EMAIL_PASSWORD = "your_app_password_here"  # Use App Password
TO_EMAIL = "hariram12003@gmail.com"

DETECTION_FRAMES_DIR = "detection_frames"
os.makedirs(DETECTION_FRAMES_DIR, exist_ok=True)

MIN_REQUIRED_CONSECUTIVE_DETECTIONS = 6
VIOLENCE_THRESHOLD = 0.55
MOTION_THRESHOLD = 3.5
HARASSMENT_CONFIDENCE_THRESHOLD = 0.55
# -------------------------------------------------------------- #

# Load VGG16 for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Custom classifier model for harassment detection
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(7 * 7 * 512,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load violence detection model
try:
    violence_model = keras.models.load_model(VIOLENCE_MODEL_PATH)
    print("\u2705 Violence detection model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load violence model: {e}")

# Load labels for harassment detection
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

# Frame preprocessing for harassment detection
def preprocess_frame(frame):
    resized = cv2.resize(frame, (224, 224))
    norm = resized.astype(np.float32) / 255.0
    return preprocess_input(np.expand_dims(norm, axis=0))

# Extract features using VGG16 before passing to the custom model
def extract_features(frame):
    preprocessed_frame = preprocess_frame(frame)
    features = base_model.predict(preprocessed_frame)
    features_flattened = features.flatten()
    return np.expand_dims(features_flattened, axis=0)

# Video Capture for Live Webcam Detection
def process_live_webcam():
    camera = cv2.VideoCapture(0)
    frame_count = 0
    consecutive_positive_frames = 0
    snapshot_taken = False

    if not camera.isOpened():
        print("❌ Error: Cannot access webcam!")
        return

    print("📷 Live monitoring started. Press ESC to exit.")

    prev_gray = None

    while True:
        ret, frame = camera.read()
        if not ret:
            print("❌ Failed to capture frame.")
            break

        frame_count += 1

        # HARASSMENT DETECTION
        h_input = extract_features(frame)
        h_pred = model.predict(h_input, verbose=0)
        h_index = np.argmax(h_pred)
        h_label = harassment_labels[h_index]
        h_score = h_pred[0][h_index]

        if h_label == "0-Positive" and h_score > HARASSMENT_CONFIDENCE_THRESHOLD:
            consecutive_positive_frames += 1
        else:
            consecutive_positive_frames = 0

        harassment_detected = consecutive_positive_frames >= MIN_REQUIRED_CONSECUTIVE_DETECTIONS

        print(f"[HARASSMENT] Class: {h_label} | Confidence: {round(h_score*100)}% | Streak: {consecutive_positive_frames}")

        # VIOLENCE DETECTION
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_magnitude = 0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motion_magnitude = np.mean(np.abs(flow))
        prev_gray = gray.copy()

        violence_input = cv2.resize(frame, (128, 128)) / 255.0
        v_pred = violence_model.predict(np.expand_dims(violence_input, axis=0), verbose=0)
        v_score = v_pred[0][0]
        violence_detected = v_score > VIOLENCE_THRESHOLD and motion_magnitude > MOTION_THRESHOLD

        print(f"[VIOLENCE] Conf: {round(v_score * 100)}% | Motion: {motion_magnitude:.2f}")

        # ALERT
        if (harassment_detected or violence_detected) and not snapshot_taken:
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
            snapshot_taken = True

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

        if cv2.waitKey(1) & 0xFF == 27:
            print("🛑 Stopping live monitoring...")
            break

    camera.release()
    cv2.destroyAllWindows()

# Start live webcam monitoring
process_live_webcam()
