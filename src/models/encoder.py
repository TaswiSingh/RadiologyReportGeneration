import torch
import torch.nn as nn
import torchvision.models as models

class VisualEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(VisualEncoder, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
