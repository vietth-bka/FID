__author__ = "vietth5, datnt527"

import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import sys

sys.path.append(os.path.dirname(__file__))

class CustomEmbedding():
    def __init__(self, device='cuda:0'):
        """
        :param device:
        """
        CHECKPOINT_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), 'BEST_checkpoint_r50.tar'))
        assert os.path.exists(CHECKPOINT_PATH)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location = "cuda:0")
        model = checkpoint['model'].module.cuda()
        model.eval()
        self.model = model
        self.data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.device = device

    def get_embedding(self, img):
        """Get normalized embedding, as well with the origin norm
        :param img: input image
        :type img: numpy.array
        :returns tuple (normalized_embedding, origin_norm)
        """
        # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = Image.fromarray(img).convert('RGB')

        rgb = self.data_transforms(rgb)
        rgb = torch.from_numpy(np.expand_dims(rgb, axis=0)).to(self.device)
        emb = self.model(rgb)
        emb = emb.detach().cpu().numpy()
        emb = np.squeeze(emb)
        norm = np.linalg.norm(emb)
        return emb / norm, norm


if __name__ == "__main__":
    test_img_path = '/home/dat/Pictures/test/1586230631.605031.jpg'
    # test_img_path = '/media/dat/05C830EB6380925C/data/faces/CBCNV_true/extracted_face/000541_Trinh Mai Quy.jpg'
    # test_img_path = '/home/dat/Pictures/10.61.166.6.png'
    # test_img_path = '/home/dat/Pictures/IMG_4993.jpg'

    cv2image = cv2.imread(test_img_path)
    # cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
    # cv2image = Image.fromarray(cv2image).convert('RGB')

    # Imageimg = Image.open(test_img_path).convert('RGB')
    Imageimg = Image.open(test_img_path)
    Imageimg = np.array(Imageimg)
    Imageimg = Imageimg[:,:,::-1]
    # Imageimg = np.transpose(np.array(Imageimg), (2,0,1))

    print('cv2image shape', cv2image.shape)
    print('Imageimg shape', Imageimg.shape)
    # print("max error", np.max(np.array(cv2image) - np.array(Imageimg)))
    cv2.imshow('cv2 image', cv2image)
    cv2.imshow('PIL image', Imageimg)
    cv2.imshow('diff', cv2image - Imageimg)
    cv2.waitKey(0)
    print("max error", np.max(cv2image - Imageimg))
    raise exit(0)


    custom_embedding = CustomEmbedding()
    test_img = cv2.imread(test_img_path)
    em, norm = custom_embedding.get_embedding(test_img)

    # the origin lines come from vietth
    CHECKPOINT_PATH='BEST_checkpoint_r50.tar'
    checkpoint = torch.load(CHECKPOINT_PATH, map_location = "cuda:0")
    model = checkpoint['model'].module.cuda()
    model.eval()

    device = 'cuda:0'
    data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def get_emb(img_path):
        img = Image.open(img_path).convert('RGB')
        img = data_transforms(img)
        img = torch.from_numpy(np.expand_dims(img, axis=0)).to(device)
        emb = model(img)
        emb = emb.detach().cpu().numpy()
        emb = np.squeeze(emb)
        return emb / np.linalg.norm(emb)
    
    viet_em = get_emb(test_img_path)

    print('em', em.shape)
    print('viet_em', viet_em.shape)
    print('max_error', np.max(em - viet_em))