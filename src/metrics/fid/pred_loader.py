import random

import torch
from torch.utils.data import Dataset

from cfg.cfg import Config

class LoadPred(Dataset):
    def __init__(self, config: Config, model, size = 256):
        super(LoadPred, self).__init__()
        self.config = config
        self.model = model
        self.size = size

    def __getitem__(self, index):
        inputs = torch.randn([1, self.config.input_dims]).to(self.config.device)
        label = torch.zeros([1, 10]).to(self.config.device)
        label[:, random.randint(0, 9)] = 1.0
        image = self.model(inputs, label)[0]
        return image, label[0]


    def __len__(self):
        return self.size