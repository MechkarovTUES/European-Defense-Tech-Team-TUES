import matplotlib.colors
import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt

max_size = 244


def reformat_box(box):
    x = (float(box['xtl']))
    y = (float(box['ytl']))

    w = abs((float(box['xtl'])) - (float(box['xbr'])))
    h = abs((float(box['ytl'])) - (float(box['ybr'])))

    return {'x': x, 'y': y, 'width': w, 'height': h}


def format_image(img, box):
    height, width = img.shape
    height_ratio = height / max_size
    width_ratio = width / max_size

    new_width = int(width / width_ratio)
    new_height = int(height / height_ratio)

    new_size = (new_width, new_height)
    resized = cv.resize(img, new_size, interpolation=cv.INTER_LINEAR)
    new_image = np.zeros((max_size, max_size), dtype=np.uint8)
    new_image[0:new_height, 0:new_width] = resized

    if box:
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        new_box = {'x': int((x / width_ratio)), 'y': int(y / height_ratio)
            , 'width': int(w / width_ratio), 'height': int(h / height_ratio)}
    else:
        new_box = None

    return new_image, new_box


def display_dataset_entries(dataset, num_entries=20):
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    axes = axes.flatten()

    for idx, ax in enumerate(axes[:num_entries]):
        if idx >= len(dataset):
            break

        entry = dataset[idx]
        img = entry.image
        bbox = entry.bounding_box

        # Display the image
        ax.imshow(img, cmap='gray')

        # Draw the bounding box
        rect = plt.Rectangle((bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                             linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Set the title
        ax.set_title(f"Entry {idx + 1}")
        ax.axis('off')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def _tensorize_dataset(dataset):
    images = []
    boxes = []

    img_size = dataset[0].image.shape[:2]

    for entry in dataset:
        img = entry.image.astype(float) / 255.
        images.append(img)

        if entry.bounding_box:
            bbox_array = [entry.bounding_box['x'], entry.bounding_box['y'], entry.bounding_box['width'],
                          entry.bounding_box['height']]

            bounding_box = np.asarray(bbox_array, dtype=float) / img_size[0]
            boxes.append(np.append(bounding_box, 1))
        else:
            boxes.append(np.zeros(5))

    X = np.array(images)

    X = np.expand_dims(X, axis=3)

    X = tf.convert_to_tensor(X, dtype=tf.float32)

    Y = tf.convert_to_tensor(boxes, dtype=tf.float32)

    result = tf.data.Dataset.from_tensor_slices((X, Y))

    return result


def tensorize_training_dataset(dataset):
    return tune_training_ds(_tensorize_dataset(dataset))


def tensorize_validation_dataset(dataset):
    return tune_validation_ds(_tensorize_dataset(dataset))

def tensorize_test_dataset(dataset):
    return tune_test_ds(_tensorize_dataset(dataset))


def format_instance(image, label):
    CLASSES = 2

    return image, (tf.one_hot(int(label[4]), CLASSES), [label[0], label[1], label[2], label[3]])



def tune_training_ds(dataset):
    BATCH_SIZE = 32

    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    dataset = dataset.repeat() # The dataset be repeated indefinitely.
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def tune_validation_ds(dataset, batch_size=64):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset


def tune_test_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(1)
    dataset = dataset.repeat()
    return dataset