import os
import torch
from torchvision import transforms
import numpy as np
import cv2
import sys
from .models import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200

def preprocess(img: np.ndarray, device="cuda:0") -> torch.Tensor:
    """preprocess image for inference"""
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    img.div_(255).sub_(0.5).div_(0.5)
    return img

class FaceFeature():
    """FaceFeature class"""
    def __init__(self, device='cuda:0'):
        checkpoint_path=os.path.abspath(os.path.join(os.path.dirname(__file__), 'backbone.pth'))
        assert os.path.exists(checkpoint_path)
        # NOTE: the checkpoint requires models.py file to be in PATH
        sys.path.append(os.path.dirname(__file__))
        checkpoint = torch.load(checkpoint_path, map_location = device)
        model = iresnet50().to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        self.model = model
        self.device = device

    def get(self, img):
        """Get normalized embedding, as well with the origin norm
        :param img: input image
        :type img: numpy.array
        :returns tuple (normalized_embedding, origin_norm)
        """
        preprocessed = preprocess(img, device=self.device)
        emb = self.model(preprocessed)
        emb = emb.detach().cpu().numpy()
        emb = np.squeeze(emb)
        norm = np.linalg.norm(emb)
        return emb / norm, norm

if __name__ == "__main__":
    test_img_path = os.path.abspath('/home/batman/Desktop/laymau_remake/src/feature/example1.jpg')
    cv2image = cv2.imread(test_img_path)

    # TODO: check cv2.imread vs Image.open
    # featurer = FaceFeature()
    # em, norm = featurer.get(cv2image)
    # print(em, norm)