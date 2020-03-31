import torch.nn as nn
from torchvision import models
from utils import get_category_list
nb_epochs = 10
learning_rate = 0.001
momentum = 0.9
device = "cuda:0"

class FlowerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet Model
        self.model = models.resnet18(pretrained=True)

        dim_before_fc = self.model.fc.in_features
        
        self.model.fc = nn.Sequential(nn.Linear(dim_before_fc, len(get_category_list())), nn.LogSoftmax())

        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def forward(self, x):
        return self.model.forward(x)