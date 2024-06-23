
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
    get_normal_image("20240621_0222.JPG", "20240621_0222_normal.JPG")