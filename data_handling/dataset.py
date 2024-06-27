import torch

from torch.utils.data import Dataset
import numpy as np

from data_handling.dataframe import Dataframe


class MineDataset(Dataset):
    def __init__(self, dataframe: Dataframe, transforms=None):
        super().__init__()

        self.dataframe = dataframe
        self.transforms = transforms

    def __getitem__(self, index: int):
        entry = self.dataframe[index]

        image = entry.image.astype(np.float32)

        boxes = np.asarray(entry.pascal_voc_bounding_boxes)

        target = {}
        if boxes.any():
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            area = torch.as_tensor(area, dtype=torch.float32)

            # there is only one class
            labels = torch.ones((boxes.shape[0]), dtype=torch.int64)
            target['boxes'] = boxes
            target['area'] = area

        else:
            labels = torch.zeros(1, dtype=torch.int64)
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['area'] = torch.zeros(1, dtype=torch.float32)

        target['labels'] = labels

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0]), dtype=torch.int64)

        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['iscrowd'] = iscrowd

        if self.transforms:
            if type(target['boxes']) is not torch.Tensor:
                sample = {
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                }
                sample = self.transforms(**sample)
                image = sample['image']

                target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

            else:
                sample = {
                    'image': image,
                    'labels': labels
                }
                sample = self.transforms(**sample)
                image = sample['image']

        return image, target

    def __len__(self) -> int:
        return self.dataframe.size
