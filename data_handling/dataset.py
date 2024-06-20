import os
import xml.etree.ElementTree as ET
import cv2 as cv

from data_handling.dataset_image import Image
from preprocessing import reformat_box, format_image


class Dataset:
    def __init__(self, dataset_folder: str = None, dataset_entries: dict = None, image_size=244):
        """

        :param dataset_entries: {'entry_name': [folder_with_images, annotation_file]}
        """
        if dataset_folder is not None:
            self.entries = self._load_folder(dataset_folder)
        elif dataset_entries is not None:
            self.entries = dataset_entries
        else:
            raise ValueError('No dataset folder or entries provided')

        self.data = self._preload_dataset()
        self.image_size = image_size

    @property
    def images(self):
        images = []
        for entry in self.entries:
            images.extend(self.data[entry])

        return images

    def _load_folder(self, folder):
        def find_annotations(folder, files):
            for file in files:
                if file.endswith('.xml'):
                    if file == 'annotations.xml' or file == f'{folder}.xml':
                        return file

            return None

        entries = {}


        # walk the folder and record every xml file and the folder it is located in
        for root1, e_dirs, _ in os.walk(folder):
            for dir in e_dirs:
                for root2, dirs, files in os.walk(root1 + dir):
                    if find_annotations(dir, files):
                        entries[dir] = [root2 + '/', root2 + '/' + find_annotations(dir, files)]
                        break
                    else:
                        break

        return entries

    def _preload_dataset(self):
        data = {}
        for entry in self.entries:
            entry_xml = self.entries[entry][1]
            data[entry] = self._preload_entry(entry_xml)

        return data

    def _load_dataset(self):
        for entry in self.data:
            entry_folder = self.entries[entry][0]
            self._load_images(entry, entry_folder)

    def _preload_entry(self, xml_file):
        img_list = []

        tree = ET.parse(xml_file)
        root = tree.getroot()

        for entry in root.iter('image'):

            filename = entry.attrib["name"]
            resolution = (entry.attrib["width"], entry.attrib["height"])

            try:
                bounding_box = entry[0].attrib
            except IndexError:
                bounding_box = None

            img_list.append(Image(filename, resolution, bounding_box))

        return img_list

    def _load_images(self, entry, entry_folder):
        def sanitaze_entry_folder(entry_folder):
            if entry_folder[-1] == '/':
                return entry_folder
            else:
                return entry_folder + '/'

        for i in self.data[entry]:
            img = cv.imread(sanitaze_entry_folder(entry_folder) + i.image_path, cv.IMREAD_GRAYSCALE)

            if i.bounding_box:
                img_box = reformat_box(i.bounding_box)
                i.bounding_box = img_box
            else:
                img_box = None

            img, img_box = format_image(img, img_box, input_size=self.image_size)
            i.bounding_box = img_box

            i.load_image(img)
