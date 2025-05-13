import cv2
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
import tensorflow as tf4
from tensorflow import keras
from collections import deque
from ultralytics import YOLO

# === Optional beep support (Windows only)
try:
    import winsound
    beep_supported = True
except ImportError:
    beep_supported = False

# === File Paths ===
HARASSMENT_MODEL_PATH = r"C:\sheild-master\best_harassment_model.pth"
VIOLENCE_MODEL_PATH = r"C:\Real-Time-Violence-Detection-main\Violence Detection\modelnew.h5"
GENDER_MODEL_PATH = r"C:\sheild-master\CCTV Gender Classifier Dataset\gender_classifier.pth"
YOLO_MODEL_PATH = "yolov8n.pt"

# === Harassment Detection Model (PyTorch MobileNetV3-Large) ===
class HarassmentModel(nn.Module):
    def __init__(self):
        super(HarassmentModel, self).__init__()
        self.model = mobilenet_v3_large(pretrained=False)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 2)

    def forward(self, x):
        return self.model(x)

harassment_model = HarassmentModel()
try:
    harassment_model.load_state_dict(torch.load(HARASSMENT_MODEL_PATH, map_location='cpu'))
    harassment_model.eval()
    print("✅ Harassment model (PyTorch) loaded.")
except Exception as e:
    print(f"❌ Error loading harassment model: {e}")

harassment_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Gender Classifier ===
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.model = mobilenet_v3_large(pretrained=False)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 2)

    def forward(self, x):
        return self.model(x)

gender_model = GenderClassifier()
try:
    state_dict = torch.load(GENDER_MODEL_PATH, map_location='cpu')
    gender_model.load_state_dict(state_dict, strict=False)
    gender_model.eval()
    print("✅ Gender model loaded.")
except Exception as e:
    print(f"❌ Error loading gender model: {e}")

gender_transform = harassment_transform
GENDER_LABELS = ['Male', 'Female']

# === Violence Detection Model ===
try:
    violence_model = keras.models.load_model(VIOLENCE_MODEL_PATH)
    print("✅ Violence model loaded.")
except Exception as e:
    print(f"❌ Error loading violence model: {e}")
    exit()

# === YOLOv8 ===
yolo_model = YOLO(YOLO_MODEL_PATH)

# === Constants ===
VIOLENCE_IMG_SIZE = (128, 128)
HARASSMENT_THRESHOLD = 0.75
VIOLENCE_THRESHOLD = 0.50
CONFIDENCE_GAP = 0.20
MOTION_THRESHOLD = 1.70
BUFFER_LENGTH = 10
MIN_HARASSMENT_FRAMES = 4
MIN_VIOLENCE_FRAMES = 1
ALERT_INTERVAL = 1

# === Webcam ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

harassment_buffer = deque(maxlen=BUFFER_LENGTH)
violence_buffer = deque(maxlen=BUFFER_LENGTH)
prev_gray = None
last_snapshot_time = 0
consecutive_harassment_count = 0

SNAPSHOT_DIR = "detection_frames"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# === Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === Motion Detection ===
    motion_magnitude = 0
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        motion_magnitude = np.mean(np.abs(flow))
    prev_gray = gray.copy()

    # === YOLO Detection ===
    results = yolo_model(frame, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    male_count = 0
    female_count = 0
    face_count = 0

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if int(classes[i]) != 0 or confidences[i] < 0.5:
            continue
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        try:
            face_tensor = gender_transform(person_crop).unsqueeze(0)
            with torch.no_grad():
                output = gender_model(face_tensor)
                gender_idx = torch.argmax(output, dim=1).item()
                gender = GENDER_LABELS[gender_idx]
        except:
            gender = "Unknown"

        color = (255, 0, 255) if gender == "Female" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if gender == "Male": male_count += 1
        if gender == "Female": female_count += 1
        face_count += 1

    # === Harassment Detection ===
    harassment_score = 0
    harassment_label = "none"
    is_harassment_frame = False

    if motion_magnitude > MOTION_THRESHOLD and face_count > 1 and male_count > 0 and female_count > 0:
        input_tensor = harassment_transform(frame).unsqueeze(0)
        with torch.no_grad():
            output = harassment_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            harassment_score = probs[0][1].item()  # Assume 1 = positive
            if harassment_score > HARASSMENT_THRESHOLD:
                is_harassment_frame = True
                consecutive_harassment_count += 1
            else:
                consecutive_harassment_count = max(0, consecutive_harassment_count - 1)
    else:
        consecutive_harassment_count = 0

    harassment_buffer.append(1 if consecutive_harassment_count >= 2 else 0)

    # === Violence Detection ===
    violence_input = np.expand_dims(cv2.resize(frame, VIOLENCE_IMG_SIZE) / 255.0, axis=0)
    violence_score = violence_model.predict(violence_input, verbose=0)[0][0]
    is_violent_frame = violence_score > VIOLENCE_THRESHOLD and motion_magnitude > MOTION_THRESHOLD
    violence_buffer.append(1 if is_violent_frame else 0)

    # === Incident Confirmation ===
    harassment_detected = sum(harassment_buffer) >= MIN_HARASSMENT_FRAMES
    violence_detected = sum(violence_buffer) >= MIN_VIOLENCE_FRAMES
    current_time = time.time()

    if harassment_detected or violence_detected:
        label = "⚠️ HARASSMENT DETECTED" if harassment_detected else "⚠️ VIOLENCE DETECTED"
        color = (0, 0, 255)
        if current_time - last_snapshot_time > ALERT_INTERVAL:
            snapshot_path = os.path.join(SNAPSHOT_DIR, f"detection_{int(current_time)}.jpg")
            cv2.imwrite(snapshot_path, frame)
            print(f"🔴 Incident confirmed! Snapshot saved: {snapshot_path}")
            if beep_supported:
                try:
                    winsound.Beep(1000, 500)
                except:
                    pass
            last_snapshot_time = current_time
    else:
        label = "✅ SAFE ENVIRONMENT"
        color = (0, 255, 0)

    # === Display ===
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Harassment: {harassment_score:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"Violence: {violence_score:.2f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"Motion: {motion_magnitude:.2f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"Faces: {face_count} | M: {male_count} F: {female_count}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow("Live Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
