import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def calculate_iou(test_entry: tf.Tensor, pred: tf.Tensor, input_size=244):
    _, labels = test_entry

    predicted_box = pred[1][0] * input_size
    predicted_box = tf.cast(predicted_box, tf.int32)

    actual_box = labels[1][0] * input_size
    actual_box = tf.cast(actual_box, tf.int32)

    IoU = intersection_over_union(predicted_box.numpy(), actual_box.numpy())

    return IoU

def visualise_test(test_entry: tf.Tensor, pred: tf.Tensor, input_size=244):
    image, labels = test_entry

    predicted_box = pred[1][0] * input_size
    predicted_box = tf.cast(predicted_box, tf.int32)

    predicted_label = pred[0][0]

    image = image[0]

    actual_label = labels[0][0]
    actual_box = labels[1][0] * input_size
    actual_box = tf.cast(actual_box, tf.int32)

    image = image.numpy().astype("float") * 255.0
    image = image.astype(np.uint8)
    image_color = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    color = (255, 0, 0)
    if (predicted_label[0] > 0.5 and actual_label[0] > 0) or (predicted_label[0] < 0.5 and actual_label[0] == 0):
        color = (0, 255, 0)

    img_label = "mine"
    if predicted_label[0] > 0.5:
        img_label = "no_mine"

    predicted_box_n = predicted_box.numpy()
    cv.rectangle(image_color, predicted_box_n, color, 2)
    cv.rectangle(image_color, actual_box.numpy(), (0, 0, 255), 2)
    cv.rectangle(image_color, (predicted_box_n[0], predicted_box_n[1] + predicted_box_n[3] - 20),
                 (predicted_box_n[0] + predicted_box_n[2], predicted_box_n[1] + predicted_box_n[3]), color, -1)
    cv.putText(image_color, img_label, (predicted_box_n[0] + 5, predicted_box_n[1] + predicted_box_n[3] - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))

    IoU = intersection_over_union(predicted_box.numpy(), actual_box.numpy())

    plt.title("IoU:" + format(IoU, '.4f'))
    plt.imshow(image_color)
    plt.show()

    return IoU
