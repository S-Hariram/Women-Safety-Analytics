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

