import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from collections import deque
from ultralytics import YOLO  # YOLOv8 for face detection

# ✅ Define Classes
CLASSES = ["FEMALE", "MALE"]
NUM_CLASSES = len(CLASSES)

# ✅ Load YOLOv8 Face Detector (Instead of Full Body)
yolo_model = YOLO("C:\\sheild-master\\yolov8n.pt")  # YOLOv8 model trained for face detection

# ✅ Load Gender Classifier Model
class GenderClassifier(nn.Module):
    def __init__(self, num_classes):  # <-- Double underscores
        super(GenderClassifier, self).__init__()  # <-- Double underscores
        self.mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.mobilenet.classifier[3] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.mobilenet(x)


# ✅ Load Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenderClassifier(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(r"C:\\sheild-master\\CCTV Gender Classifier Dataset\\gender_classifier.pth", map_location=device))
model.eval()

# ✅ Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Store Gender Predictions for Stability
prediction_history = {}  # Stores last 10 predictions for each detected person

# ✅ Open Webcam
cap = cv2.VideoCapture(0)  # 0 for laptop webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Detect Faces with YOLO
    results = yolo_model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
    confidences = results[0].boxes.conf.cpu().numpy()  # Get confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy()  # Get class ids

    male_count = 0
    female_count = 0

    # ✅ Process Each Detected Face
    for idx, (x1, y1, x2, y2) in enumerate(detections):
        conf = confidences[idx]
        class_id = int(class_ids[idx])  # Get the class ID for each detected object

        # ✅ Only process faces (class_id == 0, which is person)
        if conf < 0.6 or class_id != 0:  # Skip if confidence is low or not a person
            continue

        # ✅ Ignore Very Small Detections (Likely Hands or Noise)
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            continue

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        face_roi = frame[y1:y2, x1:x2]  # Crop the detected face

        # ✅ Convert to PIL & Transform
        img_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        # ✅ Make Prediction
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            gender = CLASSES[predicted.item()]

        # ✅ Use Rolling Average for Stability
        if idx not in prediction_history:
            prediction_history[idx] = deque(maxlen=10)

        prediction_history[idx].append(gender)
        stable_gender = max(set(prediction_history[idx]), key=prediction_history[idx].count)

        # ✅ Count Males & Females
        if stable_gender == "MALE":
            male_count += 1
            label_color = (0, 255, 0)  # Green
        else:
            female_count += 1
            label_color = (255, 0, 255)  # Pink

        # ✅ Draw Bounding Box & Label
        cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)
        cv2.putText(frame, stable_gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

    # ✅ Display Male & Female Count
    cv2.putText(frame, f"Males: {male_count} | Females: {female_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # ✅ Show Webcam Output
    cv2.imshow("Face-Based Multi-Person Gender Detection", frame)

    # ✅ Exit on 'q' Key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release Resources
cap.release()
cv2.destroyAllWindows()
