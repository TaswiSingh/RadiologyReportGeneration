import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class VisualEncoder(nn.Module):

    def __init__(
        self,
        d_model=512,
        pretrained=True,
        freeze_backbone=True
    ):
        super().__init__()

        # Load pretrained DenseNet-121
        if pretrained:
            weights = DenseNet121_Weights.DEFAULT
        else:
            weights = None

        backbone = densenet121(weights=weights)

        # Keep only the CNN feature extractor
        self.backbone = backbone.features

        # DenseNet-121 output channels
        self.feature_dim = 1024

        # Convert CNN feature dimension to Transformer dimension
        self.projection = nn.Linear(
            self.feature_dim,
            d_model
        )

        self.norm = nn.LayerNorm(d_model)

        # Freeze DenseNet initially
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images):

        # Input:
        # [B, 3, 224, 224]

        features = self.backbone(images)

        # DenseNet feature map:
        # [B, 1024, 7, 7]

        features = torch.relu(features)

        # Convert spatial feature map into visual tokens

        # [B, 1024, 7, 7]
        #       ↓
        # [B, 1024, 49]
        features = features.flatten(2)

        # [B, 1024, 49]
        #       ↓
        # [B, 49, 1024]
        features = features.transpose(1, 2)

        # Project:
        # [B, 49, 1024]
        #       ↓
        # [B, 49, 512]
        features = self.projection(features)

        features = self.norm(features)

        return features