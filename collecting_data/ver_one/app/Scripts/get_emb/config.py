import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
image_w = 112
image_h = 112
channel = 3
emb_size = 512

# Training parameters
num_workers = 1  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 200  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
num_classes = 462 #93431
num_samples =  35110 #5179510
dev_samples = 13783

pickle_train = 'data/face_417_cleanv4full_clean.pickle'
pickle_dev = 'data/face_416_cleanv4full_noise.pickle'
# pickle_train = 'data/data_180k.pickle'
# pickle_train = 'data/data_180k_50_train.pickle'
# pickle_dev = 'data/data_180k_50_dev.pickle'