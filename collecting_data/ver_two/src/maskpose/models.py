#%%
import torch
from torch import nn
from torchvision.models import resnet50
import math
import torchvision
# from bottleneck_transformer_pytorch import BottleStack
import torch.utils.model_zoo as model_zoo
# from hopenet import SEBottleneck
# from torchsummary import summary
# layer = BottleStack(
#     # dim = 256,
#     dim = 1024,
#     # fmap_size = 56,        # set specifically for imagenet's 224 x 224
#     fmap_size=7,
#     dim_out = 2048,
#     proj_factor = 4,
#     downsample = False,
#     heads = 4,
#     dim_head = 128,
#     rel_pos_emb = True,
#     activation = nn.ReLU()
# )
class MergeResNet(nn.Module):
    # BoTNet50 for regression of 3 Euler angles.
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(MergeResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        # self.BOT = layer
        self.AdaptiveAvg = nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten = nn.Flatten(1)
        self.linear = nn.Linear(512 * block.expansion,  128 * block.expansion)
        self.fc_yaw = nn.Linear(128 * block.expansion, 62)
        self.fc_pitch = nn.Linear(128 * block.expansion, 62)
        self.fc_roll = nn.Linear(128 * block.expansion, 62)

        self.fc_mask = nn.Linear(512 * block.expansion, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        # print(x.shape)
        
        # x = self.BOT(x)
        x = self.AdaptiveAvg(x)
        
        x = self.Flatten(x)
        
        x_pose = self.linear(x)
        x_pose = self.relu(x_pose)
        # print(x.shape)

        x_pose = x_pose.view(x_pose.size(0), -1)
        pre_yaw = self.fc_yaw(x_pose)
        pre_pitch = self.fc_pitch(x_pose)
        pre_roll = self.fc_roll(x_pose)

        pre_mask = self.fc_mask(x)

        
        return pre_mask, pre_yaw, pre_pitch, pre_roll
class ResNet(nn.Module):
    # BoTNet50 for regression of 3 Euler angles.
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        # self.BOT = layer
        self.AdaptiveAvg = nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten = nn.Flatten(1)
        self.linear = nn.Linear(512 * block.expansion,  128 * block.expansion)
        self.fc_yaw = nn.Linear(128 * block.expansion, 62)
        self.fc_pitch = nn.Linear(128 * block.expansion, 62)
        self.fc_roll = nn.Linear(128 * block.expansion, 62)

        # self.linear = nn.Linear(2048, 512)
        # self.fc_yaw = nn.Linear(512, 62)
        # self.fc_pitch = nn.Linear(512, 62)
        # self.fc_roll = nn.Linear(512, 62)
        # self.linear = nn.Linear(512, 128)
        # self.fc_yaw = nn.Linear(128, 62)
        # self.fc_pitch = nn.Linear(128, 62)
        # self.fc_roll = nn.Linear(128, 62)
        # self.fc_yaw = nn.Linear(512, 62)
        # self.fc_pitch = nn.Linear(512, 62)
        # self.fc_roll = nn.Linear(512, 62)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        # print(x.shape)
        
        # x = self.BOT(x)
        x = self.AdaptiveAvg(x)
        
        x = self.Flatten(x)
        x = self.linear(x)
        x = self.relu(x)
        # print(x.shape)

        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll

class FaceMask_ResNet(nn.Module):
    # BoTNet50 for regression of 3 Euler angles.
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(FaceMask_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        # self.BOT = layer
        self.AdaptiveAvg = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(512*block.expansion, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.AdaptiveAvg(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
# class BoTNet50(nn.Module):
#     # BoTNet50 for regression of 3 Euler angles.
#     def __init__(self, block, layers, num_classes=1000):
#         self.inplanes = 64
#         super(BoTNet50, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(SEBottleneck, 64, layers[0])
#         self.layer2 = self._make_layer(SEBottleneck, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(SEBottleneck, 256, layers[2], stride=2)
#         # self.layer4 = self._make_layer(SEBottleneck, 512, layers[3], stride=2)
#         # self.avgpool = nn.AvgPool2d(7)
#         self.BOT = layer
#         self.AdaptiveAvg = nn.AdaptiveAvgPool2d((1, 1))
#         self.Flatten = nn.Flatten(1)
#         # self.linear = nn.Linear(2048, 512 * block.expansion)
#         # self.fc_yaw = nn.Linear(512 * block.expansion, 62)
#         # self.fc_pitch = nn.Linear(512 * block.expansion, 62)
#         # self.fc_roll = nn.Linear(512 * block.expansion, 62)

#         # self.linear = nn.Linear(2048, 512)
#         # self.fc_yaw = nn.Linear(512, 62)
#         # self.fc_pitch = nn.Linear(512, 62)
#         # self.fc_roll = nn.Linear(512, 62)
#         self.fc_yaw = nn.Linear(512, 62)
#         self.fc_pitch = nn.Linear(512, 62)
#         self.fc_roll = nn.Linear(512, 62)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)

#         # x = self.layer4(x)
#         # x = self.avgpool(x)
#         # print(x.shape)
        
#         x = self.BOT(x)
#         x = self.AdaptiveAvg(x)
        
#         x = self.Flatten(x)
#         x = self.linear(x)
#         # print(x.shape)

#         x = x.view(x.size(0), -1)
#         pre_yaw = self.fc_yaw(x)
#         pre_pitch = self.fc_pitch(x)
#         pre_roll = self.fc_roll(x)

#         return pre_yaw, pre_pitch, pre_roll


resnet = ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
# botnet = BoTNet50(SEBottleneck, [3, 4, 23, 3])

# model surgery
# resnet = resnet50()
# backbone = list(resnet.children())

# model = nn.Sequential(
#     *backbone[:-3],
#     layer,
#     nn.AdaptiveAvgPool2d((1, 1)),
#     nn.Flatten(1),
#     nn.Linear(2048, 512),
# )

if __name__ == '__main__':
    def load_filtered_state_dict(model, snapshot):
        # By user apaszke from discuss.pytorch.org
        model_dict = model.state_dict()
        snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
        model_dict.update(snapshot)
        model.load_state_dict(model_dict)
        summary(model, (3, 112, 112))
    load_filtered_state_dict(resnet, model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
    
    # img = torch.randn(2, 3, 112, 112)
    # pre_yaw, pre_pitch, pre_roll = resnet(img) # (2, 1000)
    # print(pre_yaw.shape)