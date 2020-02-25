# AllConvNet(
#   (conv1): Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (conv2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (maxPool3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (conv4): Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (conv5): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (maxPool6): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (conv7): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (conv8): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
#   (class_conv): Conv2d(192, 10, kernel_size=(1, 1), stride=(1, 1))
# )
import torch.nn as nn
import torch.nn.functional as F

class AllConvNet(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 5, padding=1)
        self.maxPool3=nn.MaxPool2d(3,stride=2,padding=1)
        self.conv4 = nn.Conv2d(96, 192, 5, padding=1)
        self.maxPool6=nn.MaxPool2d(3,stride=2,padding=1)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)
		self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        maxPool3_out = F.relu(self.maxPool3(conv1_out))
        maxPool3_out_drop = F.dropout(maxPool3_out, .5)
        conv4_out = F.relu(self.conv4(maxPool3_out_drop))
        maxPool6_out = F.relu(self.maxPool6(conv4_out))
        maxPool6_out_drop = F.dropout(maxPool6_out, .5)
        conv7_out = F.relu(self.conv7(maxPool6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out

trial=AllConvNet(3)
print(trial)