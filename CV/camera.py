import cv2

# Open the video device
cap = cv2.VideoCapture('/dev/video3', cv2.CAP_V4L2)

if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

for i in range(4):
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    # Display the resulting frame
    cv2.imwrite(f'thermal/image_{i}.jpg', frame)
cap.release()
