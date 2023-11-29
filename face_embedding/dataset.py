# import numbers
import os
# import queue as Queue
# import threading
import pickle
import cv2
import skimage.io as io

# import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class InferFace(Dataset):
    def __init__(self, samples, reader='cv'):
        super(InferFace, self).__init__()        
        self.samples = samples
        self.reader =reader

    def __getitem__(self, index):        
        sample = self.samples[index]['img']
        label = self.samples[index]['label']

        if self.reader == 'cv':
            img = cv2.imread(sample)
        elif self.reader == 'io':
            img = io.imread(sample)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        img.div_(255).sub_(0.5).div_(0.5)
            
        return img, label, sample.split('/')[-2]+'/'+sample.split('/')[-1]

    def __len__(self):
        return len(self.samples)
