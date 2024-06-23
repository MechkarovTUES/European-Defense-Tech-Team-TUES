import os
import random
import xml.etree.ElementTree as ET
import cv2 as cv

from data_handling.dataset_image import Image
#from preprocessing import (format_image, display_dataset_entries)
# from preprocessing import (_tensorize_dataset, tensorize_training_dataset, \
#     tensorize_validation_dataset, tensorize_test_dataset, visualise_tensorised)


class Dataframe:
    def __init__(self, dataset_folder: str = None, dataset_entries: dict = None, image_size=(244, 244), random_seed=0, existing_data=None):
        """

        :param dataset_entries: {'entry_name': [folder_with_images, annotation_file]}
        """
        if existing_data is not None:
            self.data = existing_data['data']
            self.entries = existing_data['entries']
            self.root_folder = existing_data['root_folder']
            self.image_size = existing_data['image_size']
            self.random_seed = existing_data['random_seed']
            self.flatten_data = self.images

        else:
            if dataset_folder is not None:
                self.root_folder = dataset_folder
                self.entries = self._load_folder(dataset_folder)
            elif dataset_entries is not None:
                self.entries = dataset_entries
            else:
                raise ValueError('No dataset folder or entries provided')

            self.data = self._preload_dataset()

            self.flatten_data = self.images

            self.image_size = image_size
            self.random_seed = random_seed

    def __getitem__(self, index: int) -> Image:
        return self.flatten_data[index]

    def __len__(self):
        return len(self.flatten_data)

    @property
    def images(self):
        images = []
        for entry in self.entries:
            images.extend(self.data[entry])

        return images

    @property
    def size(self):
        return len(self.images)

    def load_images(self):
        self._load_dataset()

    def train_test_split(self, train_ratio=70, validation_ratio=20):
        dataset = self.images

        train_end = int(len(self.images) * (train_ratio / 100))
        validation_end = train_end + int(len(self.images) * (validation_ratio / 100))

        random.seed(self.random_seed)

        random.shuffle(dataset)

        train_set = dataset[:train_end]
        validation_set = dataset[train_end:validation_end]
        test_set = dataset[validation_end:]

        train_data = {'data': {}, 'entries': {}, 'root_folder': self.root_folder, 'image_size': self.image_size, 'random_seed': self.random_seed}
        validation_data = {'data': {}, 'entries': {}, 'root_folder': self.root_folder, 'image_size': self.image_size, 'random_seed': self.random_seed}
        test_data = {'data': {}, 'entries': {}, 'root_folder': self.root_folder, 'image_size': self.image_size, 'random_seed': self.random_seed}

        for img in train_set:
            for entry in self.entries:
                if img in self.data[entry]:
                    if entry not in train_data['data']:
                        train_data['data'][entry] = []
                    train_data['data'][entry].append(img)
                    if entry not in train_data:
                        train_data['entries'][entry] = self.entries[entry]
                    break

        for img in validation_set:
            for entry in self.entries:
                if img in self.data[entry]:
                    if entry not in validation_data['data']:
                        validation_data['data'][entry] = []
                    validation_data['data'][entry].append(img)
                    if entry not in validation_data:
                        validation_data['entries'][entry] = self.entries[entry]
                    break

        for img in test_set:
            for entry in self.entries:
                if img in self.data[entry]:
                    if entry not in test_data['data']:
                        test_data['data'][entry] = []
                    test_data['data'][entry].append(img)
                    if entry not in test_data:
                        test_data['entries'][entry] = self.entries[entry]
                    break


        return Dataframe(existing_data=train_data), Dataframe(existing_data=validation_data), Dataframe(existing_data=test_data)


#   def tensorize_dataset(self):
#       return _tensorize_dataset(self.images)
#
#   def tensorized_train_test_split(self):
#       train, validation, test = self.train_test_split()
#
#       return tensorize_training_dataset(train), tensorize_validation_dataset(validation), tensorize_test_dataset(test)

#    def preview_dataset(self, sample=None, count=32):
#        if count > len(self.images):
#            count = len(self.images)
#
#        if sample is None:
#            sample = random.sample(self.images, count)
#
#        display_dataset_entries(sample)

#    def preview_tensorised(self, ts_dataset, count=32):
#        visualise_tensorised(ts_dataset, input_size=self.image_size, count=count)

    def reseed(self, new_random_seed):
        self.random_seed = new_random_seed

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
            self._load_images(entry, self.root_folder)

    def _preload_entry(self, xml_file):
        img_list = []

        tree = ET.parse(xml_file)
        root = tree.getroot()

        for entry in root.iter('image'):

            filename = entry.attrib["name"]
            resolution = (entry.attrib["width"], entry.attrib["height"])

            if entry.attrib['id'] == '57':
                print(entry.attrib['id'])

            try:
                bounding_boxes = [b.attrib for b in entry.iter('box')]
            except IndexError:
                bounding_boxes = None

            img_list.append(Image(filename, resolution, bounding_boxes))

        return img_list

    def _load_images(self, entry, root_folder):
        for i in self.data[entry]:
            img = cv.imread(root_folder + i.image_path, cv.IMREAD_GRAYSCALE)

#           new_img, new_img_boxes = format_image(img, i.coco_bounding_boxes, input_size=self.image_size)
#           for k in range(len(i.bounding_boxes)):
#               i.bounding_boxes[k]['xtl'] = new_img_boxes[k][0]
#               i.bounding_boxes[k]['ytl'] = new_img_boxes[k][1]
#               i.bounding_boxes[k]['xbr'] = new_img_boxes[k][0] + new_img_boxes[k][2]
#               i.bounding_boxes[k]['ybr'] = new_img_boxes[k][1] - new_img_boxes[k][3]

            i.load_image(img)
