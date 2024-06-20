from data_handling.dataset import Dataset

if __name__ == "__main__":
    dataset = Dataset(dataset_folder='new_output/')
    print(dataset.images)
