import torch
import torch.nn as nn


class MammogramClassifier(nn.Module):
    def __init__(self, n_classes=3):
        super(MammogramClassifier, self).__init__()

        self.n_classes = n_classes
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.model.fc = nn.Linear(512, self.n_classes)

    def forward(self, x):
        x = self.model(x)
        return x
