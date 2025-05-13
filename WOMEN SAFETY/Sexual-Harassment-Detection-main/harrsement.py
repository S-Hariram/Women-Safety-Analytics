import cv2
import numpy as np
import smtplib
import threading
import os
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import img_to_array
from collections import deque
import datetime

# Email Configuration
EMAIL_ADDRESS = "your_email@gmail.com"  # Your email address
EMAIL_PASSWORD = "your_app_password"  # App password for Gmail
TO_EMAIL = "recipient_email@gmail.com"  # Recipient email

# Load VGG16 for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Custom classifier model
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(7 * 7 * 512,)))  # Flattened VGG16 output
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # Binary classification (harassment or not)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Dummy loading for weights (adjust this with the actual trained model weights)
# model.load_weights("path_to_your_trained_model_weights.h5")  # Load your trained model weights

# Moving Average Buffer for Prediction Smoothing
prediction_buffer = deque(maxlen=10)

def send_email(image_path, current_time):
    """Send email with snapshot attached"""
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

        # Connect to Gmail server and send the email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, TO_EMAIL, msg.as_string())
        server.quit()

        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Error while sending email: {e}")

def preprocess_frame(frame):
    """Preprocess the captured frame"""
    frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frame_array = img_to_array(frame_resized)
    frame_preprocessed = preprocess_input(frame_array)
    return frame_preprocessed

def predict_frame(frame):
    """Make a prediction for the given frame"""
    preprocessed_frame = preprocess_frame(frame)
    features = base_model.predict(np.expand_dims(preprocessed_frame, axis=0))
    features = features.reshape(1, 7 * 7 * 512)  # Flatten the output
    prediction = model.predict(features)
    predicted_class_index = np.argmax(prediction[0])
    return predicted_class_index, prediction[0][predicted_class_index]

def process_live_webcam():
    """Processes live webcam frames for harassment detection"""
    camera = cv2.VideoCapture(0)  # Webcam stream
    alert_threshold = 5  # Number of detections needed to confirm harassment
    min_required_detections = 4 # Consecutive detections needed for a valid alert
    frame_skip = 2  # Skip frames to speed up detection
    frame_count = 0
    positive_count = 0
    snapshot_taken = False
    image_output_path = r"C:\WOMEN SAFETY\Sexual-Harassment-Detection-main\snapshot.jpg"

    if not camera.isOpened():
        print("❌ Error: Cannot access webcam!")
        return

    print("📷 Live monitoring started. Press ESC to exit.")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            print("❌ Failed to capture frame.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Predict harassment in the current frame
        predicted_class, confidence = predict_frame(frame)

        # Update the prediction buffer for smoothing
        prediction_buffer.append(confidence if predicted_class == 1 else 0)

        # Check for consecutive positive detections (harassment detected)
        if sum(1 for score in list(prediction_buffer)[-8:] if score > 0.5) >= min_required_detections:
            positive_count += 1
        else:
            positive_count = 0
            snapshot_taken = False

        # Capture snapshot and send email after 4 consecutive detections
        if positive_count == min_required_detections and not snapshot_taken:
            print("📸 Capturing snapshot (4th detection)...")
            if cv2.imwrite(image_output_path, frame):
                print("✅ Snapshot saved successfully!")
                snapshot_taken = True

        # Send email after 5 detections
        if positive_count >= alert_threshold and not snapshot_taken:
            print("🚨 Harassment Confirmed! Sending email...")
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            email_thread = threading.Thread(target=send_email, args=(image_output_path, current_time))
            email_thread.start()
            break  # Exit after sending the email

        # Display the live frame with prediction label
        cv2.putText(frame, f"Prediction: {'Harassment' if predicted_class == 1 else 'Safe'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Harassment Detection", frame)

        # Exit the loop if the 'ESC' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            print("🛑 Stopping live monitoring...")
            break

    camera.release()
    cv2.destroyAllWindows()

# Start live detection
process_live_webcam()
 