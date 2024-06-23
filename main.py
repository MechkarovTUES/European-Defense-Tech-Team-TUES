import time

from model.model import compile_model, plot_accuracy, plot_classification_loss, plot_regression_loss
from data_handling import Dataframe, MineDataset

from test import visualise_test, calculate_iou

if __name__ == "__main__":
    dataframe = Dataframe(dataset_folder='new_output/')
    dataframe.load_images()

    train, val, test = dataframe.train_test_split()

    train_dataset = MineDataset(train)
    val_dataset = MineDataset(val)
    test_dataset = MineDataset(test)

    t = train_dataset.__getitem__(190)
    print(f"Dataset has {dataframe.size} images")

    time.sleep(5)

    print("Commencing computer torture...")

    test = None
