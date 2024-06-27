import time

from data_handling import Dataframe, MineDataset


if __name__ == "__main__":
    dataframe = Dataframe(dataset_folder="munich_dataset")
    dataframe.load_images()

    train, val, test = dataframe.train_test_split()

    print(f"Dataset has {dataframe.size} images")

