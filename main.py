import time
import tensorflow as tf

from model.model import compile_model, plot_accuracy, plot_classification_loss, plot_regression_loss
from data_handling.dataset import Dataset

from test import visualise_test, calculate_iou

if __name__ == "__main__":
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())

    dataset = Dataset(dataset_folder='new_output/')
    print(f"Dataset has {dataset.size} images")

    time.sleep(5)

    print("Commencing computer torture...")

    dataset.load_images()

    model = compile_model()
    model.save_weights('model.h5')

    test = None

    avg_iou = 0
    while avg_iou < 0.2:
        dataset.reseed(time.time())
        train, val, test = dataset.tensorized_train_test_split()

        model.load_weights('model.h5')

        EPOCHS = 120
        BATCH_SIZE = 64

        history = model.fit(train,
                            steps_per_epoch=(dataset.train_end // BATCH_SIZE),
                            validation_data=val, validation_steps=3,
                            epochs=EPOCHS, verbose="false")

        tests = test.take(20)
        avg_iou = 0
        for test_entry in tests:


            res = model(test_entry[0])

            avg_iou += calculate_iou(test_entry, res)

        avg_iou /= 20

        print(f"Average IoU is {avg_iou}")

    avg_iou = 0
    for test_entry in test.take(20):
        res = model(test_entry[0])

        avg_iou += visualise_test(test_entry, res)

    avg_iou /= 20

    print(f"Average IoU is {avg_iou}")