"""
ResNet18-backed Road Defect Detection Network
Uses ImageNet-pretrained ResNet18 backbone with a custom dual head
for Defect Classification and Spatial Bounding Box Regression.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class RoadDefectResNet18(nn.Module):
    """
    Transfer learning network built on ResNet18 for high-precision
    road defect detection (potholes and cracks) on real pavement imagery.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(RoadDefectResNet18, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet18(weights=weights)

        # Retain feature extractor up to avgpool
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool

        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.fc_cls = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # Spatial Bounding Box Regression Head [cx, cy, w, h] normalized in [0, 1]
        self.fc_bbox = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feat = torch.flatten(x, 1)

        cls_logits = self.fc_cls(feat)
        bbox_pred = self.fc_bbox(feat)
        return cls_logits, bbox_pred

# Backward compatible alias
RoadDefectCNN = RoadDefectResNet18
