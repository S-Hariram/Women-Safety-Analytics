import cv2

for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Testing camera {i}")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(3000)  # Show for 3 seconds
        cap.release()
        cv2.destroyAllWindows()
