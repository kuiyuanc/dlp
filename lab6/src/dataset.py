import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Normalize, Resize, ToTensor


class iclevrDataset(Dataset):
    def __init__(self, root, mode="train"):
        super().__init__()

        assert mode in ["train", "test", "new_test"], "mode should be either 'train', 'test', or 'new_test'"

        with open(f"{mode}.json", "r") as json_file:
            json_data = json.load(json_file)
            if mode == "train":
                self.image_paths, labels = tuple(json_data.keys()), tuple(json_data.values())
            else:
                labels = json_data
            self.len = len(labels)

        with open("objects.json", "r") as json_file:
            objects = json.load(json_file)

        self.labels_one_hot = torch.zeros(self.len, len(objects))
        for i, label in enumerate(labels):
            self.labels_one_hot[i][[objects[j] for j in label]] = 1

        self.root = root
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        label_one_hot = self.labels_one_hot[index]
        if self.mode == "train":
            image = self._load_image(index)
            image = iclevrDataset.transform_image(image)
            return image, label_one_hot
        else:
            return label_one_hot

    def _load_image(self, index):
        image_path = Path(self.root, self.image_paths[index])
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def transform_image(image):
        transform = transforms.Compose([Resize((64, 64)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return transform(image)
