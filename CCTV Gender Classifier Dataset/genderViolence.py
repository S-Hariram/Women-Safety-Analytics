import cv2
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from collections import deque
from PIL import Image
import winsound
import tensorflow as tf
from tensorflow import keras
from ultralytics import YOLO
import smtplib
from email.message import EmailMessage

# === Constants and Paths ===
CLASSES = ["FEMALE", "MALE"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = (128, 128)
VIOLENCE_THRESHOLD = 0.4
MOTION_THRESHOLD = 1.2
VIOLENCE_CONFIRM_COUNT = 1
ALERT_INTERVAL = 2  # seconds
MODEL_PATH = r"C:\Real-Time-Violence-Detection-main\Violence Detection\modelnew.h5"
VIOLENCE_FRAMES_DIR = "violence_frames"
os.makedirs(VIOLENCE_FRAMES_DIR, exist_ok=True)

# === Email Settings ===

EMAIL_ADDRESS = "Your gmail address"
EMAIL_PASSWORD = "password"  # Use App Password
TO_EMAIL = "sender gmail address"

def send_email_alert(image_path):
    try:
        msg = EmailMessage()
        msg['Subject'] = '⚠️ Violence Detected Alert'
        msg['From'] = EMAIL_ADDRESS
        msg['To'] =TO_EMAIL
        msg.set_content('Violence has been detected. Please find the attached snapshot.')

        with open(image_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(image_path)
            msg.add_attachment(file_data, maintype='image', subtype='jpeg', filename=file_name)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"[EMAIL] Alert sent to {TO_EMAIL}")
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send email: {e}")

# === Load Models ===
yolo_model = YOLO("C:\\sheild-master\\yolov8n.pt")

class GenderClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GenderClassifier, self).__init__()
        self.mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.mobilenet.classifier[3] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gender_model = GenderClassifier(num_classes=NUM_CLASSES).to(device)
gender_model.load_state_dict(torch.load(r"C:\\sheild-master\\CCTV Gender Classifier Dataset\\gender_classifier.pth", map_location=device))
gender_model.eval()

# === TensorFlow Violence Model ===
violence_model = keras.models.load_model(MODEL_PATH)

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Runtime Variables ===
cap = cv2.VideoCapture(0)
prev_gray = None
last_alert_time = 0
violence_counter = 0
alert_active = False
violenceCount = [0, int(time.time())]
danger_shown = False
danger_start_time = 0
prediction_history = {}

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


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ===== GENDER DETECTION =====
    results = yolo_model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    male_count = 0
    female_count = 0

    for idx, (x1, y1, x2, y2) in enumerate(detections):
        conf = confidences[idx]
        class_id = int(class_ids[idx])

        if conf < 0.6 or class_id != 0 or (x2 - x1) < 50 or (y2 - y1) < 50:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        face_roi = frame[y1:y2, x1:x2]
        img_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = gender_model(img_tensor)
            _, predicted = torch.max(output, 1)
            gender = CLASSES[predicted.item()]

        if idx not in prediction_history:
            prediction_history[idx] = deque(maxlen=10)
        prediction_history[idx].append(gender)
        stable_gender = max(set(prediction_history[idx]), key=prediction_history[idx].count)

        label_color = (0, 255, 0) if stable_gender == "MALE" else (255, 0, 255)
        if stable_gender == "MALE":
            male_count += 1
        else:
            female_count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)
        cv2.putText(frame, stable_gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

    # ===== VIOLENCE DETECTION =====
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
            alert_active = True # === HARASSMENT CONDITION ===
            if male_count == 1 and female_count == 1:
                incident_type = "HARASSMENT"
                print("⚠️ Harassment Detected")
            else:
                incident_type = "VIOLENCE"
                print("⚠️ Violence Detected")

            send_email_alert(filepath)

    # === Display Overlay ===
    if violence_counter >= VIOLENCE_CONFIRM_COUNT:
        if male_count >= 1 and female_count >= 1:
            label = "⚠️ HARASSMENT DETECTED"
            color = (147, 20, 255)
        else:
            label = "⚠️ VIOLENCE DETECTED"
            color = (0, 0, 255)
    else:
        label = "✅ NO VIOLENCE"
        color = (0, 255, 0)
            

    # === Display ===
   

    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Model Score: {prediction:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"Motion: {motion_magnitude:.2f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # === DANGER ALERT ON SCREEN ===
    if danger_shown and current_time - danger_start_time <= 5:
        cv2.putText(frame, "🚨 DANGER 🚨", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    elif current_time - danger_start_time > 5:
        danger_shown = False


    cv2.putText(frame, f"Males: {male_count} | Females: {female_count}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    
    cv2.imshow("Gender + Violence Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
