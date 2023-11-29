import os
import argparse
import torch
import cv2
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pytorch_Retinaface'))

from .data import cfg_mnet, cfg_re50
from .layers.functions.prior_box import PriorBox
from .utils.nms.py_cpu_nms import py_cpu_nms
from .models.retinaface import RetinaFace
from .utils.box_utils import decode, decode_landm

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class FaceDetector:
    def __init__(self) -> None:
        torch.set_grad_enabled(False)
        trained_model = os.path.join(os.path.dirname(__file__), 'weights/Resnet50_Final.pth')
        cfg = cfg_re50
        # cfg = cfg_mnet

        # net and model
        net = RetinaFace(cfg=cfg, phase = 'test')
        net = load_model(net, trained_model, False)
        net.eval()
        device = torch.device("cuda")
        net = net.to(device)       

        self.net = net
        self.device = device
        self.cfg = cfg

    def detect(self, frame: np.ndarray, threshold: float = 0.8, scale: float = 1):
        """
        @return bboxes
        @return landmarks
        """
        img = np.float32(frame)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)  # forward pass

        # postprocessing
        resize = 1
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        confidence_threshold = 0.02
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        top_k = 5000
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        nms_threshold = 0.4
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        keep_top_k = 750
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        bboxes = []
        landmarks = []
        for b in dets:
            # b[4] is the prob
            # (b[0], b[1]), (b[2], b[3]) is the bbox
            # b[5:15] is the landmarks
            if b[4] < threshold:
                continue

            bboxes.append(b[0:5])
            landmarks.append(b[5::].reshape(5, 2))
        return bboxes, landmarks
            

if __name__ == '__main__':
    faceDetector = FaceDetector()

    image_path = "/home/dat/source/faceid/facedetection_trt/test/worlds-largest-selfie.jpg"
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    bboxes, landmarks = faceDetector.detect(img_raw, threshold=0.8)

    # show image
    for bbox, landmark in zip(bboxes, landmarks):
        if bbox[4] < 0.8:
            continue
        text = "{:.4f}".format(bbox[4])
        b = list(map(int, bbox))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(img_raw, (landmark[0][0], landmark[0][1]), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (landmark[1][0], landmark[1][1]), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (landmark[2][0], landmark[2][1]), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (landmark[3][0], landmark[3][1]), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (landmark[4][0], landmark[4][1]), 1, (255, 0, 0), 4)

    cv2.imshow('detection_result', img_raw)
    cv2.waitKey(0)

