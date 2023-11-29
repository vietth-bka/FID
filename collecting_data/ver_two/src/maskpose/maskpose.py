"""MaskPose class, use trained model in python code"""
import os
import cv2
import numpy as np
import torch
from torchvision.models.resnet import BasicBlock
from .models import MergeResNet
# from models import MergeResNet

def preprocess(img: np.ndarray, mean: float = 127.5, std: float = 128.0, device="cuda:0") -> torch.Tensor:
    """preprocess image for inference"""
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32)
    img = (img - mean) / std
    img = np.expand_dims(img, axis=0)
    tensor = torch.from_numpy(img).to(device)
    return tensor

def softmax_temperature(tensor, temperature):
    """just softmax temperature"""
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result

def load_filtered_state_dict(model, state_dict):
    """By user apaszke from discuss.pytorch.org"""
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

class MaskPose:
    """MaskPose class"""

    def __init__(self, device="cuda:0"):
        self.device = device
        model = MergeResNet(BasicBlock, [2, 2, 2, 2])
        model.to(device)
        snapshot = os.path.join(os.path.dirname(__file__), "MaskPose_R18_310721.pkl")
        state_dict = torch.load(snapshot, map_location=device)
        load_filtered_state_dict(model, state_dict)
        model.eval()
        self.model = model

        idx_tensor = list(range(62))
        self.idx_tensor = torch.Tensor(idx_tensor).to(device)

    def get_pose(self, image: np.ndarray) -> np.ndarray:
        """get pose from image
        Args:
            image: image in BGR format
        """
        # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # no need
        preprocessed = preprocess(image, device=self.device)

        mask, yaw, pitch, roll = self.model(preprocessed)
        yaw = softmax_temperature(yaw.data, 1)
        yaw = torch.sum(yaw * self.idx_tensor, 1).to(self.device) * 3 - 93

        pitch = softmax_temperature(pitch.data, 1)
        pitch = torch.sum(pitch * self.idx_tensor, 1).to(self.device) * 3 - 93

        roll = softmax_temperature(roll.data, 1)
        roll = torch.sum(roll * self.idx_tensor, 1).to(self.device) * 3 - 93

        return torch.argmax(mask).item(), yaw.item(), pitch.item(), roll.item()

if __name__ == "__main__":
    test_image = "example1.jpg"
    test_text = test_image.replace("jpg", "txt")
    test_image = cv2.imread(test_image)

    with open(test_text) as f:
        lines = f.readlines()
        is_mask, yaw, pitch, roll = (float(i.rstrip()) for i in lines)
        print("WHENET masked =", is_mask, yaw, pitch, roll)

    MaskPose = MaskPose()
    masked, yaw, pitch, roll = MaskPose.get_pose(test_image)

    print(masked, yaw, pitch, roll)