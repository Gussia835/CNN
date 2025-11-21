import os
import json
import torch
from PIL import Image
import torch.nn as nn
import torch.utils.data as data 


class Dataset(data.Dataset):

    def __init__(self, 
                 path,
                 file_trainning, 
                 train=True, 
                 transform=None):
        self.img_dir = os.path.join(path, 'train' if train else 'label')
        self.transform = transform

        with open(os.path.join(path, file_trainning), 'r') as fp:
            file_extension = file_trainning.split('.')[1]
            match file_extension:
                case 'json':
                    self.format = json.load(fp)
                # case 'xml':
                # case 'Yolo':

        self.length = 0
        self.files = []
        self.targets = torch.eye(10)

        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)
            self.length += len(list_files)
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target),
                                  list_files))

        def __getitem__(self, item):
            img_path, target = self.files[item]
            t = self.targets[target]
            img = Image.open(img_path)
            return img, t

        def __len__(self):
            return self.length