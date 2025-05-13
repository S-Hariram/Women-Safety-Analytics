import cv2
import numpy as np
import datetime
import threading
import smtplib
import os
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from geopy.geocoders import Nominatim
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from collections import deque

np.set_printoptions(suppress=True)

# Load Model & Labels
MODEL_PATH = r"C:\sheild-master\model\keras_model.h5"
LABELS_PATH = r"C:\sheild-master\model\labels.txt"

model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Email Configuration (Use App Password for security)
EMAIL_ADDRESS = "hariramsundaram@gmail.com"
EMAIL_PASSWORD = "jywr idqd spma mumu"  # Replace with actual App Password
TO_EMAIL = "hariram12003@gmail.com"

# Moving Average Buffer for Prediction Smoothing
prediction_buffer = deque(maxlen=10)


def send_email(image_path, current_time, location_details):
    """Sends an email with an alert and snapshot."""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_EMAIL
        msg['Subject'] = "⚠️ Harassment Alert Detected"

        body = f"🚨 Alert!\nTime: {current_time}\nLocation: {location_details}"
        msg.attach(MIMEText(body, 'plain'))

        if os.path.exists(image_path):
            with open(image_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(image_path)}")
                msg.attach(part)
            print(f"📎 Attaching {image_path} to email.")
        else:
            print("❌ Snapshot file not found! Sending email without an image.")

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, TO_EMAIL, msg.as_string())
        server.quit()

        print("✅ Email sent successfully!")
    except smtplib.SMTPAuthenticationError:
        print("❌ Gmail Authentication failed! Use an App Password.")
    except Exception as e:
        print(f"❌ Error while sending email: {e}")


def preprocess_frame(frame):
    """Preprocesses a frame for the model."""
    frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frame_preprocessed = np.asarray(frame_resized, dtype=np.float32) / 255.0
    frame_preprocessed = np.expand_dims(frame_preprocessed, axis=0)
    return frame_preprocessed


def predict_frame(frame):
    """Predicts harassment probability from a frame."""
    preprocessed_frame = preprocess_frame(frame)
    prediction = model.predict(preprocessed_frame)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score


def get_location():
    """Fetches the current location using geopy."""
    geolocator = Nominatim(user_agent="SHEild/1.0 (karanjot032004@gmail.com)")
    try:
        location = geolocator.geocode("12A State Highway, CGC Jhanjeri, Mohali, Punjab, India")
        return f"{location.address}, Lat: {location.latitude}, Lon: {location.longitude}" if location else "Location not found"
    except Exception as e:
        return f"Error retrieving location: {e}"


def process_video(video_path):
    """Processes video frames and detects harassment."""
    camera = cv2.VideoCapture(video_path)
    alert_threshold = 10  # Email alert trigger
    min_required_detections = 8  # Snapshot at 8 consecutive detections
    frame_skip = 3
    frame_count = 0
    positive_count = 0
    snapshot_taken = False
    image_output_path = r"C:\sheild-master\alert_snapshot.jpg"

    if not camera.isOpened():
        print("❌ Error: Cannot open video file!")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            print("❌ End of video or failed to capture frame.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        class_name, confidence_score = predict_frame(frame)
        prediction_buffer.append(confidence_score if class_name == "0-Positive" else 0)

        weights = np.linspace(1, 2, len(prediction_buffer))
        avg_confidence = np.average(prediction_buffer, weights=weights)
        confidence_threshold = max(0.5, np.mean(prediction_buffer) - 0.05)

        print(f"Class: {class_name} | Confidence: {round(confidence_score * 100)}% (Avg: {round(avg_confidence * 100)}%)")

        if sum(1 for val in list(prediction_buffer)[-8:] if val > confidence_threshold) >= min_required_detections:
            positive_count += 1
        else:
            positive_count = 0
            snapshot_taken = False  

        if positive_count == min_required_detections and not snapshot_taken:
            print("📸 Capturing snapshot (8th detection)...")
            if cv2.imwrite(image_output_path, frame):
                print("✅ Snapshot saved successfully!")
                snapshot_taken = True  
            else:
                print("❌ Failed to save snapshot!")

        if positive_count >= alert_threshold:
            print("🚨 Harassment Confirmed (10th detection)! Sending email...")
            time.sleep(1)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            location_details = get_location()
            email_thread = threading.Thread(target=send_email, args=(image_output_path, current_time, location_details))
            email_thread.start()
            break  

        cv2.imshow("Live Monitoring", frame)
        if cv2.waitKey(1) == 27:
            print("🛑 Stopping video processing...")
            break

    camera.release()
    cv2.destroyAllWindows()


def process_image(image_path):
    """Processes an image and detects harassment."""
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Failed to load image.")
        return

    class_name, confidence_score = predict_frame(frame)
    print(f"Class: {class_name} | Confidence: {round(confidence_score * 100)}%")
    
    if class_name == "0-Positive" and confidence_score > 0.7:
        print("🚨 Harassment Detected in Image! Sending alert...")
        image_output_path = "alert_snapshot.jpg"
        if cv2.imwrite(image_output_path, frame):
            print("✅ Snapshot saved successfully!")
        else:
            print("❌ Failed to save snapshot!")

        time.sleep(1)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        location_details = get_location()
        email_thread = threading.Thread(target=send_email, args=(image_output_path, current_time, location_details))
        email_thread.start()


# Choose Mode (Image or Video)
mode = "video"  # Change to "image" for image processing
if mode == "image":
    process_image(r"C:\sheild-master\postive\_108362648_sexualharassment976.jpg")
else:
    process_video(r"C:\sheild-master\vi.mp4")
