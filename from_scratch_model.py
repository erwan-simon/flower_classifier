import torch.nn as nn
import torch.nn.functional as F
from utils import get_category_list


nb_epochs = 50
learning_rate = 0.00025
momentum = 0.9
device = "cuda:0"

class FlowerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # image shape is 200 x 200 x 3
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        
        # self.lin1 = nn.Linear(64, 100)
        # self.lin2 = nn.Linear(1000, 100)
        self.lin3 = nn.Linear(64, len(get_category_list()))
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(batch_size, 64, -1).mean(dim=2)
        # x = F.relu(self.lin1(x))
        # x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x