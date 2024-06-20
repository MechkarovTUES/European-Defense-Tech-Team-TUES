class Image:
    def __init__(self, image_path, resolution: tuple, bounding_box):
        self.image = None
        self.image_path = image_path
        self.resolution = resolution
        self.bounding_box = bounding_box

    def load_image(self, image):
        self.image = image

    def get_img_name(self):
        return self.image_path.split('/')[-1]