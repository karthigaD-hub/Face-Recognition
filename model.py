import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FaceNet(nn.Module):
    def __init__(self, num_classes):
        super(FaceNet, self).__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        layers = list(base.children())[:-1]
        self.backbone = nn.Sequential(*layers)
        self.embedding = nn.Linear(2048, 512)
        self.bn = nn.BatchNorm1d(512)
        self.id_classifier = nn.Linear(512, num_classes)
        self.gender_classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        emb = self.bn(self.embedding(x))
        emb = F.normalize(emb, p=2, dim=1)
        id_out = self.id_classifier(emb)
        gender_out = self.gender_classifier(emb)
        return emb, id_out, gender_out
