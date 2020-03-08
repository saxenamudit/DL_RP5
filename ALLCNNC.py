#ALL CNN C
# AllConvNet(
#   (conv1): Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (conv2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (conv3): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#   (conv4): Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (conv5): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (conv6): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#   (conv7): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (conv8): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
#   (class_conv): Conv2d(192, 10, kernel_size=(1, 1), stride=(1, 1))
# )
import torch.nn as nn
import torch.nn.functional as F

class AllConvNet(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1) #1
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1) #2
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2) #3
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1) #4
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1) #5
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2) #6
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1) #7
        self.conv8 = nn.Conv2d(192, 192, 1) #8
        self.class_conv = nn.Conv2d(192, n_classes, 1) #9


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out

trial=AllConvNet(3)
print(trial)

# output dimesions 
# AllConvNet(
#   (net): Sequential(
#     (0): Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (2): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#     (3): Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (5): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#     (6): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
#     (8): Conv2d(192, 10, kernel_size=(1, 1), stride=(1, 1))
#   )
# )
# torch.Size([32, 96, 32, 32]) 1
# torch.Size([32, 96, 32, 32]) 2
# torch.Size([32, 96, 16, 16]) 3
# torch.Size([32, 192, 16, 16]) 4
# torch.Size([32, 192, 16, 16]) 5
# torch.Size([32, 192, 8, 8]) 6
# torch.Size([32, 192, 8, 8]) 7
# torch.Size([32, 192, 8, 8]) 8
# torch.Size([32, 10, 8, 8]) 9