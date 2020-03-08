import torch.nn as nn
import torch.nn.functional as F

class AllConvNet(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(input_size, 96, 3, padding=1),nn.Conv2d(96, 96, 3, padding=1),nn.Conv2d(96, 96, 3, padding=1, stride=2),nn.Conv2d(96, 192, 3, padding=1),nn.Conv2d(192, 192, 3, padding=1),nn.Conv2d(192, 192, 3, padding=1, stride=2),nn.Conv2d(192, 192, 3, padding=1),nn.Conv2d(192, 192, 1),nn.Conv2d(192, n_classes, 1))

    def forward(self, x):
     for layer in self.net:
      x = layer(x)
      print(x.size())
     return x

model=AllConvNet(3)
print(model)