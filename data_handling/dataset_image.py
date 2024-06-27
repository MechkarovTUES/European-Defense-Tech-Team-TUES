class Image:
    def __init__(self, image_path, resolution: tuple, bounding_boxes):
        self.image = None
        self.image_path = image_path
        self.resolution = resolution
        self.bounding_boxes = bounding_boxes

    def load_image(self, image):
        self.image = image

    def get_img_name(self):
        return self.image_path.split('/')[-1]

    @property
    def coco_bounding_boxes(self):
        bounding_boxes = []
        for box in self.bounding_boxes:
            x = (float(box['xtl']))
            y = (float(box['ytl']))

            w = abs((float(box['xtl'])) - (float(box['xbr'])))
            h = abs((float(box['ytl']) - (float(box['ybr']))))

            bounding_boxes.append([x, y, w, h])

        return bounding_boxes

    @property
    def pascal_voc_bounding_boxes(self):
        bounding_boxes = []
        for box in self.bounding_boxes:
            xmin = (float(box['xtl']))
            ymin = (float(box['ytl']))

            ymax = (float(box['ybr']))
            xmax = (float(box['xbr']))

            if ymax > float(self.resolution[1]):
                ymax = float(self.resolution[1])

            if xmax > float(self.resolution[0]):
                xmax = float(self.resolution[0])

            bounding_boxes.append([xmin, ymin, xmax, ymax])

        return bounding_boxes
