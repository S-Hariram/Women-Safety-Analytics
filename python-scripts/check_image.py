import cv2
import numpy as np
from keras.models import load_model

# Load Model
model = load_model(r"C:\\sheild-master\\model\\keras_model.h5", compile=False)

# Load Class Labels
class_names = open(r"C:\\sheild-master\\model\\labels.txt", "r").readlines()

# Load and Preprocess Image
image_path = r"C:\\sheild-master\\-ve images\\2e3015d7-efcd-4018-ab50-e91527f9c7f2\\beautiful-young-black-african-businesswoman-260nw-262457150.jpg"  # Change this to your image path
image = cv2.imread(image_path)

if image is None:
    print("❌ Error: Could not load image. Check the file path.")
    exit()

# Resize to match model input size
image_resized = cv2.resize(image, (224, 224))

# Preprocess (Normalize)
image_preprocessed = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
image_preprocessed = (image_preprocessed / 127.5) - 1  # Normalize between -1 and 1

# Make Prediction
prediction = model.predict(image_preprocessed)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]

# Display Result
print(f"🔹 Class: {class_name} | Confidence: {confidence_score * 100:.2f}%")

# Show the Image
cv2.putText(image, f"{class_name} ({confidence_score * 100:.2f}%)", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Test Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
