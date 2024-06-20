from data_handling.dataset import Dataset

if __name__ == "__main__":
    dataset = Dataset(dataset_folder='new_output/')
    print(dataset.images)
    dataset.load_images()
    print(dataset.images)
    train, val, test = dataset.tensorized_train_test_split()
    dataset.preview_tensorised(train)
    print(len(val))
