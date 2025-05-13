import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model(r"C:\\sheild-master\\model\\gender_classifier.h5")

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Resize the frame to 224x224
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to range [0, 1]
    return normalized_frame.reshape(1, 224, 224, 3)  # Reshape to (1, 224, 224, 3) for model input

def predict_gender(frame):
    input_data = preprocess_frame(frame)  # Preprocess the frame
    predictions = model.predict(input_data)  # Get model predictions
    return "male" if predictions[0][0] > predictions[0][1] else "female"  # Return predicted gender

def detect_people(frame):
    # Placeholder: Implement actual people detection (e.g., using a pre-trained model like OpenCV's Haar cascades)
    # Using pre-defined bounding boxes as an example. Replace this with a real person detection algorithm.
    return [(50, 50, 100, 200), (300, 50, 100, 200)]  # Example bounding boxes

# Read the image from disk
image = cv2.imread(r"C:\\sheild-master\\demo.jpg")  # Replace with the actual path to your image

if image is None:
    print("Error loading image!")
else:
    boxes = detect_people(image)  # Detect bounding boxes around people
    img_height, img_width, _ = image.shape
    male_count = 0
    female_count = 0

    # Process each bounding box
    for box in boxes:
        x, y, w, h = box
        if x >= 0 and y >= 0 and x + w <= img_width and y + h <= img_height:
            person_frame = image[y:y + h, x:x + w]  # Crop the region of interest (ROI) for gender detection
            gender = predict_gender(person_frame)  # Predict gender for the person in the ROI
            if gender == "male":
                male_count += 1
            else:
                female_count += 1

            # Show the image with the detected bounding boxes and gender predictions
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(image, f'Gender: {gender}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Display gender text

        else:
            print(f"Bounding box {box} is out of bounds!")
            continue  # Skip if the bounding box is out of bounds

    # Display the image using OpenCV's imshow
    cv2.imshow("Image with Gender Prediction", image)
    cv2.waitKey(0)  # Wait for any key to close the image window
    cv2.destroyAllWindows()  # Close the OpenCV window

    # Print out the total counts of males and females detected
    print(f"Total Males: {male_count}")
    print(f"Total Females: {female_count}")
