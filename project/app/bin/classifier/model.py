import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiOutputModel(nn.Module):
    def __init__(self, n_Modality_classes, n_Bodypart_classes):
        super().__init__()
        self.base_model = models.mobilenet_v2().features 
        last_channel = models.mobilenet_v2().last_channel 

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.Modality = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_Modality_classes)
        )
        self.Bodypart = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_Bodypart_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        return {
            'Modality': self.Modality(x),
            'Bodypart': self.Bodypart(x),
        }

    def get_loss(self, net_output, ground_truth):
        Modality_loss = F.cross_entropy(net_output['Modality'], ground_truth['Modality_labels'])
        Bodypart_loss = F.cross_entropy(net_output['Bodypart'], ground_truth['Bodypart_labels'])
        loss = Modality_loss + Bodypart_loss
        return loss