import cv2

origin = cv2.VideoCapture(0)
thermal = cv2.VideoCapture(3)

if not origin.isOpened():
    print("Error: Could not open video device")
    exit()

# Set resolution
origin.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
origin.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# Print some properties
# print("Frame Width: ", origin.get(cv2.CAP_PROP_FRAME_WIDTH))
# print("Frame Height: ", origin.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print("FPS: ", origin.get(cv2.CAP_PROP_FPS))
# print("Brightness: ", origin.get(cv2.CAP_PROP_BRIGHTNESS))
# print("Contrast: ", origin.get(cv2.CAP_PROP_CONTRAST))

for i in range(4):
    ret1, frameOrigin = origin.read()
    ret2, frameThermal = thermal.read()

    if not ret1 and not ret2:
        print("Error: Could not read frame")
        break

    cv2.imwrite(f'thermal/image_{i}.jpg', frameThermal)
    cv2.imwrite(f'origin/image_{i}.jpg', frameOrigin)

origin.release()
thermal.release()