# === Combined Violence and Harassment Detection ===

import cv2
import numpy as np
import smtplib
import threading
import os
import time
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from collections import deque
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from ultralytics import YOLO
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img


# === Email Configuration ===
EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
TO_EMAIL = "recipient_email@gmail.com"

# === Harassment Model ===
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    Dense(1024, activation='relu', input_shape=(7 * 7 * 512,)),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.load_weights("path_to_your_trained_model_weights.h5")  # Uncomment and load your model

# === Violence Model ===
VIOLENCE_MODEL_PATH = r"C:\Real-Time-Violence-Detection-main\Violence Detection\modelnew.h5"
violence_model = load_model(VIOLENCE_MODEL_PATH)

# === YOLO Model ===
yolo_model = YOLO("yolov8n.pt")

# === Constants ===
VIOLENCE_IMG_SIZE = (128, 128)
VIOLENCE_THRESHOLD = 0.75
MOTION_THRESHOLD = 2.5
MIN_VIOLENCE_FRAMES = 4
BUFFER_LENGTH = 10

# === Snapshot ===
SNAPSHOT_DIR = "detection_frames"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
SNAPSHOT_PATH = os.path.join(SNAPSHOT_DIR, "snapshot.jpg")

# === Helpers ===
def send_email(image_path, current_time):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_EMAIL
        msg['Subject'] = "⚠️ Harassment Alert Detected"

        body = f"🚨 Harassment detected at {current_time}."
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
        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Error while sending email: {e}")

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_array = img_to_array(frame_resized)
    frame_preprocessed = preprocess_input(frame_array)
    return frame_preprocessed

def predict_harassment(frame):
    preprocessed_frame = preprocess_frame(frame)
    features = base_model.predict(np.expand_dims(preprocessed_frame, axis=0))
    features = features.reshape(1, 7 * 7 * 512)
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction[0])
    return predicted_class, prediction[0][predicted_class]

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    prev_gray = None
    violence_buffer = deque(maxlen=BUFFER_LENGTH)
    prediction_buffer = deque(maxlen=10)
    positive_count = 0
    alert_threshold = 5
    min_required_detections = 4
    snapshot_taken = False

    print("📷 Live monitoring started. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Motion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_magnitude = 0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motion_magnitude = np.mean(np.abs(flow))
        prev_gray = gray.copy()

        # Violence Prediction
        violence_input = np.expand_dims(cv2.resize(frame, VIOLENCE_IMG_SIZE) / 255.0, axis=0)
        violence_score = violence_model.predict(violence_input, verbose=0)[0][0]
        is_violent = violence_score > VIOLENCE_THRESHOLD and motion_magnitude > MOTION_THRESHOLD
        violence_buffer.append(1 if is_violent else 0)
        violence_detected = sum(violence_buffer) >= MIN_VIOLENCE_FRAMES

        # Harassment Prediction
        predicted_class, confidence = predict_harassment(frame)
        prediction_buffer.append(confidence if predicted_class == 1 else 0)

        if sum(1 for score in list(prediction_buffer)[-8:] if score > 0.5) >= min_required_detections:
            positive_count += 1
        else:
            positive_count = 0
            snapshot_taken = False

        if positive_count == min_required_detections and not snapshot_taken:
            cv2.imwrite(SNAPSHOT_PATH, frame)
            print("✅ Snapshot saved.")
            snapshot_taken = True

        if positive_count >= alert_threshold:
            print("🚨 Harassment Confirmed! Sending email...")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            email_thread = threading.Thread(target=send_email, args=(SNAPSHOT_PATH, timestamp))
            email_thread.start()
            positive_count = 0
            snapshot_taken = False

        # Display
        label = "✅ SAFE ENVIRONMENT"
        if violence_detected:
            label = "⚠️ VIOLENCE DETECTED"
        elif predicted_class == 1 and confidence > 0.5:
            label = "⚠️ HARASSMENT DETECTED"

        color = (0, 255, 0) if label.startswith("✅") else (0, 0, 255)
        cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Live Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
