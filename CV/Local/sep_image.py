import os
import re
import cv2
import numpy as np


def image_filter(temperature_array, threshold, temp_range):
    filter_temp = np.copy(temperature_array)
    if (temp_range[1]+temp_range[0])/2 > threshold:
        filter_temp[temperature_array > threshold] = temp_range[0]
    else:
        filter_temp[temperature_array > threshold] = temp_range[1]
    return filter_temp

def recreate_image_from_temperatures(temperature_array, temp_range, output_path):
    # Normalize the temperature values to the range 0-255
    normalized_image = (temperature_array - temp_range[0]) / (temp_range[1] - temp_range[0]) * 255.0

    # Clip values to be in the valid range 0-255
    normalized_image = np.clip(normalized_image, 0, 255)

    # Convert to uint8 type
    image_uint8 = normalized_image.astype(np.uint8)

    # Save the recreated image
    cv2.imwrite(output_path, image_uint8)
    print(f"Recreated image saved to {output_path}")


def temperature_array(image_path, temp_range):
    # Read the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Image not found or unable to read.")

    height, width = image.shape
    temperature_array = np.zeros((height, width), dtype=np.float32)

    temperature_array = temp_range[0] + (image / 255.0) * (temp_range[1] - temp_range[0])
    np.set_printoptions(threshold=np.inf)
    return temperature_array


def get_files(folder):
    jpg_files = [file for file in os.listdir(folder) if file.lower().endswith('.jpg')]
    jpg_files_sorted = sorted(jpg_files, key=lambda x: int(re.search(r'_(\d+)\.', x).group(1)))
    for file in jpg_files_sorted:
        get_normal_image(f"Thermal2/{file}", f"clear/{file}")
    return len(jpg_files)


def crop_image(input_path, output_path, crop_box):
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Could not read image from {input_path}")
        return

    # Crop the image using the crop_box tuple (x, y, width, height)
    x, y, w, h = crop_box
    cropped_image = img[y:y+h, x:x+w]

    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped image saved to {output_path}")


def get_normal_image(image_path, output_path):
    def find_second_image_start(file_path, hex_pattern="ffd9"):
        pattern_bytes = bytes.fromhex(hex_pattern)

        occurrence_count = 0
        position = -1

        try:
            with open(file_path, "rb") as file:
                data = file.read()
                start = 0

                while True:
                    position = data.find(pattern_bytes, start)

                    if position == -1:
                        break

                    occurrence_count += 1

                    if occurrence_count == 4:
                        return position

                    start = position + 1

            return None
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None

    img_start = find_second_image_start(image_path)

    with open(image_path, "rb") as input_file:
        with open(output_path, "wb") as output_file:
            input_file.seek(img_start+2)
            output_file.write(b"\xff\xd8")
            output_file.write(input_file.read())


if __name__ == "__main__":
    # get_normal_image("thermal/20240622_0223.JPG", "clear/20240621_0222_normal.JPG")
    get_files("Thermal2")

    # crop_image("test/ImageTest13.jpg", "test/ImageTest_cropped.jpg", (42, 20, 71, 80))
    # temp_range = [15.4, 21.8]
    # temp = temperature_array("test/ImageTest_cropped.jpg", temp_range)
    # filter_temp = image_filter(temp, 18, temp_range)
    # recreate_image_from_temperatures(filter_temp, temp_range, "test/recreation.jpg")

