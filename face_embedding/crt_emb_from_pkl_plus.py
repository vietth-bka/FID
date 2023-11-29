import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torch import nn
import sys 

# from FaceX_Zoo.backbone.backbone_def import BackboneFactory
from dataset import InferFace

sys.path.insert(0, '../Face_KD')
from backbones import get_model
import sys
import pickle
import os
from glob import glob
from tqdm import tqdm
import random
# import fire

class FaceModel(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.
    
    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """
    def __init__(self, backbone_factory):
        """Init face model by backbone factorcy and head factory.
        
        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.adaptor = nn.Conv2d(128, 128, 1, bias=False)
        nn.init.normal_(self.adaptor.weight, 0, 0.1)
        self.backbone = backbone_factory
        self.head = nn.Linear(489, 512, bias=False)

    def forward(self, data):
        feat = self.backbone.forward(data)
        return feat

def gen_pkl(mode='ref', reader='cv', rand=False, s=5):
    """
    Generate a pickle of face embedded vector 
    """
    samples = []
    images = []

    if mode == 'ref':
        synData = '/media/v100/DATA4/thviet/Pure_dataset/synthesis'
        cnt_people = len(os.listdir(synData))
        untrained = '/media/v100/DATA4/thviet/Visiting_Data_MaskVTX'        
        print('Running untrained folders')
        for mem in os.listdir(untrained):
            if mem not in os.listdir(synData):
                cnt_people += 1
                imgs = [img for img in glob(os.path.join(untrained, mem, '*')) if '.txt' not in img and '.json' not in img]
                if rand:
                    paths = [i for i in imgs if 'ref' in i]
                    k = min(s, len(imgs))
                    if len(paths) < k:
                        tmp2 = random.sample([i for i in imgs if 'ref' not in i], k-len(paths))
                    paths.extend(tmp2) 
                else:
                    paths = imgs
                
                for img in paths:
                    samples.append({'img':img, 'label':mem})
                    images.append(img)
        print('Done for visiting')
    elif mode == 'test':
        synData = 'your/testing/data'        
        cnt_people = len(os.listdir(synData))

    for mem in os.listdir(synData):
        imgs = [img for img in glob(os.path.join(synData, mem, '*')) if '.txt' not in img and '.json' not in img]
        if rand:
            paths = [i for i in imgs if 'ref' in i]
            if len(paths) < s:
                tmp2 = random.sample([i for i in imgs if 'ref' not in i], s-len(paths))
            paths.extend(tmp2)
        else:
            paths = imgs

        for img in paths:          
            samples.append({'img':img, 'label':mem})
            images.append(img)

    print('Total samples:', len(samples))
    print('Total links:', len(images))
    print('Total people:', cnt_people)
    print('Reader:', reader)

    dataset = InferFace(samples, reader)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, num_workers=4, shuffle=False)
    device = torch.device('cuda:0')
    
    name_net = 'r50'    
    net = get_model(name_net, fp16=False).to(device)
    
    net = net.to(device)    
    std = torch.load('/media/v100/DATA4/thviet/KD_research/FaceX-Zoo/training_mode/conventional_training/out_dir/BEST_logits_mix_lr0.01_r50S_r100T_KD.pt', map_location=device)['state_dict']
    net.load_state_dict(std)
    net.eval()
    net = torch.nn.DataParallel(net).cuda()

    pkls = []
    with torch.no_grad():
        for (img, label, key) in tqdm(data_loader):
            img = img.to(device)
            feat = F.normalize(net(img)).detach().cpu().numpy()
            # feat = np.squeeze(feat)
            pkl = [{'emb':feat[i], 'name':label[i], 'key':key[i]} for i in range(len(label))]
            pkls.extend(pkl)

    if rand:
        name1 = f'rand{s}_glint_{name_net}_BGD_{cnt_people}_{mode}_{reader}.pkl'
    else:
        name1 = f'Data_pkls/glint_{name_net}_BGD_maskVTX_{cnt_people}_{mode}_{reader}.pkl'
        
    with open(name1, 'wb') as f:
        pickle.dump(pkls, f)
    
    print('Saved', name1)
    print('Size pkl:',len(pkls))

if __name__ == '__main__':
    gen_pkl('ref', 'cv', False)