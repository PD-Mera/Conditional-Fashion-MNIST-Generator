import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from cfg.cfg import Config

class LoadDataset(Dataset):
    def __init__(self, config: Config):
        super(LoadDataset, self).__init__()
        self.total_classes = config.total_classes
        self.data_path = config.data_path
        self.images = []

        if isinstance(self.data_path, str):
            self.get_images(self.data_path)
        elif isinstance(self.data_path, list):
            for path in self.data_path:
                self.get_images(path)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=config.mean, std=config.std)
        ])


    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("L")
        image = self.transform(image)
        label = int(self.images[index].split("/")[-2])
        return image, self.to_one_hot(label)


    def __len__(self):
        return len(self.images)
    

    def get_images(self, data_path):
        for subclass in os.listdir(data_path):
            subclass_path = os.path.join(data_path, subclass)
            for image in os.listdir(subclass_path):
                self.images.append(os.path.join(subclass_path, image))
    
    def to_one_hot(self, class_num: int):
        out = torch.zeros(self.total_classes)
        out[class_num] = 1.0
        return torch.FloatTensor(out)