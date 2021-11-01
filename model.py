import torch
import os
import torchvision.models as models
from torch import nn


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = None
        if cfg.model.backbone == 'resnet-18':
            self.backbone = models.resnet18()
        elif cfg.model.backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2()
        elif cfg.model.backbone == 'shufflenet_v2_x0_5':
            self.backbone = models.shufflenet_v2_x0_5()
        hidden_size = 100
        num_classes = 2
        self.classifier = nn.Sequential(
            nn.Linear(1000, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        feats = self.backbone(inputs)
        return self.classifier(feats)


def build_model(cfg):
    model = Model(cfg)
    return model
