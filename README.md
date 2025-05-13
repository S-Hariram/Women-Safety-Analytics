🚨 Real-Time Violence and Harassment Detection System
This project uses deep learning and computer vision to detect violent or harassing behavior in real-time through a live camera feed. It differentiates between general violence and harassment based on detected genders, and raises instant alerts via audio and email.

📌 Features
🔍 Real-Time Detection using TensorFlow and YOLOv8

🧍‍♂️🧍‍♀️ Gender Classification with MobileNetV3 (PyTorch)

🛑 Violence vs Harassment differentiation

📧 Email Alerts with evidence snapshots

🔊 Beep Alarms and danger overlay on screen

🎥 Works with Webcam or DroidCam (mobile camera input)

🧠 Motion detection via Optical Flow

📂 Stores frames of detected incidents

🧠 Models Used
Violence Detection Model – TensorFlow (.h5)

Gender Classifier – PyTorch (MobileNetV3)

YOLOv8 – For human detection and bounding boxes

🛠️ Tech Stack
Python, OpenCV, PyTorch, TensorFlow/Keras

YOLOv8 from Ultralytics

Email via smtplib

TorchVision transforms

PIL for image preprocessing

▶️ How to Run
Clone the repository and install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Download and place the following models in correct paths:

Violence model → modelnew.h5

Gender classifier → gender_classifier.pth

Update your email credentials in the script.

To start detection, run:

bash
Copy
Edit
python violence_detection.py
💡 Tip: If using DroidCam, make sure to use the correct camera index (e.g., cv2.VideoCapture(3)).

🖥️ Output Overview
Bounding boxes labeled MALE/FEMALE

On-screen alerts:

⚠️ VIOLENCE DETECTED

⚠️ HARASSMENT DETECTED

🚨 DANGER 🚨 (for repeated incidents)

Frame is saved and emailed during detection

📧 Email Alert
Subject: ⚠️ Violence Detected Alert

Body: Includes brief info and attached frame snapshot

Sent to configured recipient
