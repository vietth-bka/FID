"""
@author: Jun Wang
@date: 20201101
@contact: jun21wangustc@gmail.com
"""

import os
import sys
import logging as logger
import cv2
import pickle
import numpy as np
import torch

sys.path.append('../../data_processor')
from train_dataset import ultra_transform

from torch.utils.data import Dataset

logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

class CommonTestDataset(Dataset):
    """ Data processor for model evaluation.

    Attributes:
        image_root(str): root directory of test set.
        image_list_file(str): path of the image list file.
        crop_eye(bool): crop eye(upper face) as input or not.
    """
    def __init__(self, image_root, image_list_file, crop_eye=False):
        self.image_root = image_root
        self.image_list = []
        image_list_buf = open(image_list_file)
        line = image_list_buf.readline().strip()
        while line:
            self.image_list.append(line)
            line = image_list_buf.readline().strip()
        self.mean = 127.5
        self.std = 128.0
        self.crop_eye = crop_eye
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, index):
        short_image_path = self.image_list[index]
        image_path = os.path.join(self.image_root, short_image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        #image = cv2.resize(image, (128, 128))
        if self.crop_eye:
            image = image[:60, :]
        image = (image.transpose((2, 0, 1)) - self.mean) / self.std
        image = torch.from_numpy(image.astype(np.float32))
        return image, short_image_path

class TripletTest(Dataset):
    def __init__(self):
        self.test_root = '/media/v100/DATA4/thviet/KD_research/FaceX-Zoo/training_mode/conventional_training/pkls/test.pkl'
        self.data_test = pickle.load(open(self.test_root, 'rb'))
        print(self.data_test[0:10])

        self.test_labels = [d['label'] for d in self.data_test]
        self.test_data = [d['img'] for d in self.data_test]
        self.labels_set = set(self.test_labels)
        self.label_to_index = {label: np.where(np.array(self.test_labels) == label)[0] for label in self.labels_set}
        
        random_state = np.random.RandomState(29)
        
        self.test_triplets = [[i, random_state.choice(self.label_to_index[self.test_labels[i]]),
                                  random_state.choice(self.label_to_index[
                                      random_state.choice(
                                          list(self.labels_set - set([self.test_labels[i]]))
                                                      )
                                                                          ])
                              ] for i in range(len(self.test_data))]

class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, test_root):
        self.test_root = test_root
        self.test_data = pickle.load(open(self.test_root, 'rb'))

        self.test_labels = [d['label'] for d in self.test_data]
        self.test_data = [d['img'].replace('DATA2', 'DATA4').replace('vietth', 'thviet') for d in self.test_data]
        # generate fixed triplets for testing
        self.labels_set = set(self.test_labels)
        self.label_to_indices = {label: np.where(np.array(self.test_labels) == label)[0]
                                    for label in self.labels_set}

        random_state = np.random.RandomState(29)

        self.test_triplets = [[i,
                        random_state.choice(self.label_to_indices[self.test_labels[i]]),
                        random_state.choice(self.label_to_indices[
                                                random_state.choice(
                                                    list(self.labels_set - set([self.test_labels[i]]))
                                                )
                                            ])
                        ]
                    for i in range(len(self.test_data))]

    def __getitem__(self, index):
        image_path1 = self.test_data[self.test_triplets[index][0]]
        image_path2 = self.test_data[self.test_triplets[index][1]]
        image_path3 = self.test_data[self.test_triplets[index][2]]

        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
        image3 = cv2.imread(image_path3)
        
        img1 = ultra_transform(image1)
        img2 = ultra_transform(image2)
        img3 = ultra_transform(image3)

        return (img1, img2, img3)

    def __len__(self):
        return len(self.test_data)

if __name__ == '__main__':
    dataset = TripletTest()
    labels = dataset.test_labels
    print(len(labels))
    print(dataset.test_triplets[0:10])

    dataset2 = TripletMNIST()
    print(dataset2.test_triplets[0:10])