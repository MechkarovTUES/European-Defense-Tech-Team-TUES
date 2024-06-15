import random
import xml.etree.ElementTree as ET
import cv2 as cv

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from image_class import Entry
from preprocessing import reformat_box, format_image, display_dataset_entries, tensorize_training_dataset, tensorize_validation_dataset, tensorize_test_dataset

def load_dataset(xml_file: str):
    dataset = []

    tree = ET.parse(xml_file)
    root = tree.getroot()

    for entry in root.iter('image'):

        filename = entry.attrib["name"]
        resolution = (entry.attrib["width"], entry.attrib["height"])

        try:
            bounding_box = entry[0].attrib
        except IndexError:
            bounding_box = None

        dataset.append(Entry(filename, resolution, bounding_box))

    return dataset


def train_test_split(dataset, train_ratio=70, validation_ratio=20, test_ratio=10, random_seed=0):
    random.seed(random_seed)

    random.shuffle(dataset)

    t_r = int(len(dataset) * (train_ratio / 100))

    v_r = t_r + int(len(dataset) * (validation_ratio / 100))

    return dataset[:t_r], dataset[t_r:v_r], dataset[v_r:]

def load_images(dataset):
    for entry in dataset:
        img = cv.imread('new_output/' + entry.image_path, cv.IMREAD_GRAYSCALE)

        if entry.bounding_box:
            img_box = reformat_box(entry.bounding_box)
            entry.bounding_box = img_box
        else:
            img_box = None

        img, img_box = format_image(img, img_box)
        entry.bounding_box = img_box

        entry.load_image(img)


if __name__ == "__main__":
    dataset = load_dataset("annotations.xml")
    load_images(dataset)

    train_ds, val_ds, test_ds = train_test_split(dataset)

    train_ds = tensorize_training_dataset(train_ds)
    val_ds = tensorize_validation_dataset(val_ds)
    test_ds = tensorize_test_dataset(test_ds)

    print(len(val_ds))
