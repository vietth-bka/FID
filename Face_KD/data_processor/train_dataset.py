"""
@author: Hang Du, Jun Wang
@date: 20201101
@contact: jun21wangustc@gmail.com
"""

import os
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
from torchvision import transforms
import imgaug.augmenters as iaa

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def ultra_transform(image):
    """ Transform a image by cv2.
    """
    img_size = image.shape[0]
    # random crop
    rand_crop = random.random()
    if rand_crop >= 0.95:
        aug = iaa.Emboss(alpha=(0., 1.), strength=1.)
        image = aug(image=image)
    elif rand_crop >= 0.9 and rand_crop < 0.95:
        sigma_color = random.choice(np.arange(75, 100, 5))
        image = cv2.bilateralFilter(image,9,sigma_color,sigma_color)
    elif rand_crop >= 0.85 and rand_crop < 0.9:
        k = random.choice(np.arange(3, 7, 2))
        image = cv2.GaussianBlur(image,(k,k),0)
    elif rand_crop >= 0.7 and rand_crop < 0.85:
        aug = iaa.JpegCompression(compression=(50, 75))
        image = aug(image=image)
    elif rand_crop >= 0.55 and rand_crop < 0.7:
        aug = iaa.MotionBlur(k=(3, 9), angle=[-90, 90])
        image = aug(image=image)
    elif rand_crop >= 0.45 and rand_crop < 0.55:
        aug = iaa.AdditiveGaussianNoise(scale=(0.01*255, 0.03*255))
        image = aug(image=image)
        
    # adjust contrast 
    rand_coeff = random.random()
    if rand_coeff > 0.85:
        gamma = random.choice(np.arange(0.45, 0.65, 0.01))
        image = adjust_gamma(image, gamma)
    elif rand_coeff <= 0.85 and rand_coeff > 0.7:
        alpha = random.choice(np.arange(1.25, 1.55, 0.05))
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    # horizontal flipping
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    # grayscale conversion
    if random.random() > 0.9:
        image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # rotation
    if random.random() > 0.9:
        theta = (random.randint(-10,10)) * np.pi / 180
        M_rotate = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0]], dtype=np.float32)
        image = cv2.warpAffine(image, M_rotate, (img_size, img_size))
    
    # normalizing
    img_size = image.shape[0]
    if image.ndim == 2:
        image = (image - 127.5) * 0.0078125
        new_image = np.zeros([3,img_size,img_size], np.float32)
        new_image[0,:,:] = image
        new_image[1,:,:] = image
        new_image[2,:,:] = image
        image = torch.from_numpy(new_image.astype(np.float32))
    else:
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
        image = torch.from_numpy(image.astype(np.float32))

    return image

class ImageDataset(Dataset):
    def __init__(self, data_root, mode):
        self.mode = mode
        self.data_root = data_root
        self.transform = ultra_transform
        with open(self.data_root,'rb') as f:
          self.data = pickle.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sample = self.data[index]
        image_path, image_label = sample['img'], sample['label']
        image_path = image_path.replace('DATA2', 'DATA4').replace('vietth', 'thviet')
        image = cv2.imread(image_path)
        # augmentation
        if self.mode == 'train':
            image = self.transform(image)
        # normalizing
        img_size = image.shape[0]
        if image.ndim == 2:
            image = (image - 127.5) * 0.0078125
            new_image = np.zeros([3,img_size,img_size], np.float32)
            new_image[0,:,:] = image
            new_image[1,:,:] = image
            new_image[2,:,:] = image
            image = torch.from_numpy(new_image.astype(np.float32))
        else:
            image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
            image = torch.from_numpy(image.astype(np.float32))

        return image, image_label

# class ImageDataset(Dataset):
#     def __init__(self, data_root, train_file, crop_eye=False):
#         self.data_root = data_root
#         self.train_list = []
#         train_file_buf = open(train_file)
#         line = train_file_buf.readline().strip()
#         while line:
#             image_path, image_label = line.split(' ')
#             self.train_list.append((image_path, int(image_label)))
#             line = train_file_buf.readline().strip()
#         self.crop_eye = crop_eye
#     def __len__(self):
#         return len(self.train_list)
#     def __getitem__(self, index):
#         image_path, image_label = self.train_list[index]
#         image_path = os.path.join(self.data_root, image_path)
#         image = cv2.imread(image_path)
#         if self.crop_eye:
#             image = image[:60, :]
#         #image = cv2.resize(image, (128, 128)) #128 * 128
#         if random.random() > 0.5:
#             image = cv2.flip(image, 1)
#         if image.ndim == 2:
#             image = image[:, :, np.newaxis]
#         image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
#         image = torch.from_numpy(image.astype(np.float32))
#         return image, image_label

class ImageDataset_SST(Dataset):
    def __init__(self, data_root, exclude_id_set):
        """
        Load all data, create a dict whose keys are labels and values are image's path.
        """
        self.data_root = data_root
        label_set = set()
        self.id2image_path_list = {}
        with open(self.data_root, 'rb') as f:
            samples = pickle.load(f)
        
        for sample in samples:
            image_path, label = sample['img'], sample['label']
            image_path = image_path.replace('DATA2', 'DATA4').replace('vietth', 'thviet')

            if label in exclude_id_set:
                continue
            else:
                label_set.add(label)
            if not label in self.id2image_path_list:
                self.id2image_path_list[label] = []
            self.id2image_path_list[label].append(image_path)
        self.train_list = list(label_set)
        print('Valid ids: %d.' % len(self.train_list))           

    # def __init__(self, data_root, train_file, exclude_id_set):
    #     self.data_root = data_root
    #     label_set = set()
    #     # get id2image_path_list
    #     self.id2image_path_list = {}
    #     train_file_buf = open(train_file)
    #     line = train_file_buf.readline().strip()
    #     while line:
    #         image_path, label = line.split(' ')
    #         label = int(label)
    #         if label in exclude_id_set:
    #             line = train_file_buf.readline().strip()
    #             continue
    #         label_set.add(label)
    #         if not label in self.id2image_path_list:
    #             self.id2image_path_list[label] = []
    #         self.id2image_path_list[label].append(image_path)
    #         line = train_file_buf.readline().strip()
    #     self.train_list = list(label_set)
    #     print('Valid ids: %d.' % len(self.train_list))
            
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        cur_id = self.train_list[index]
        cur_image_path_list = self.id2image_path_list[cur_id]
        if len(cur_image_path_list) == 1:
            image_path1 = cur_image_path_list[0]
            image_path2 = cur_image_path_list[0]
        else:
            training_samples = random.sample(cur_image_path_list, 2)
            image_path1 = training_samples[0]
            image_path2 = training_samples[1]
        image_path1 = os.path.join(self.data_root, image_path1)
        image_path2 = os.path.join(self.data_root, image_path2)
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
        image1 = ultra_transform(image1)
        image2 = ultra_transform(image2)

        if random.random() > 0.5:
            return image2, image1, cur_id
            
        return image1, image2, cur_id

def basic_transform(image):
    """ Transform a image by cv2.
    """
    img_size = image.shape[0]
    # random crop
    if random.random() > 0.5:
        crop_size = 9
        x1_offset = np.random.randint(0, crop_size, size=1)[0]
        y1_offset = np.random.randint(0, crop_size, size=1)[0]
        x2_offset = np.random.randint(img_size-crop_size, img_size, size=1)[0]
        y2_offset = np.random.randint(img_size-crop_size, img_size, size=1)[0]
        image = image[x1_offset:x2_offset,y1_offset:y2_offset]
        image = cv2.resize(image,(img_size,img_size))
    # horizontal flipping
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    # grayscale conversion
    if random.random() > 0.8:
        image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # rotation
    if random.random() > 0.5:
        theta = (random.randint(-10,10)) * np.pi / 180
        M_rotate = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0]], dtype=np.float32)
        image = cv2.warpAffine(image, M_rotate, (img_size, img_size))
    # normalizing
    if image.ndim == 2:
        image = (image - 127.5) * 0.0078125
        new_image = np.zeros([3,img_size,img_size], np.float32)
        new_image[0,:,:] = image
        image = torch.from_numpy(new_image.astype(np.float32))
    else:
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
        image = torch.from_numpy(image.astype(np.float32))
    return image