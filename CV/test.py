import cv2
import time
# Try different indices: 0, 1, 2, etc.
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

# Set the format to MJPEG
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Print some properties
print("Frame Width: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Frame Height: ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS: ", cap.get(cv2.CAP_PROP_FPS))
print("Brightness: ", cap.get(cv2.CAP_PROP_BRIGHTNESS))
print("Contrast: ", cap.get(cv2.CAP_PROP_CONTRAST))

for i in range(4):
    ret, frame = cap.read()


    if not ret:
        print("Error: Could not read frame")
        break

    cv2.imwrite(f'thermal/image_{i}.jpg', frame)

cap.release()