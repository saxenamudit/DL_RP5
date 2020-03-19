import os
import re
allCnnC="""
class AllConvNet(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.class_conv = nn.Conv2d(192, n_classes, 1)


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
        return pool_out"""
modelC="""
class AllConvNet(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.maxPool3=nn.MaxPool2d(3,stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.maxPool6=nn.MaxPool2d(3,stride=2, padding=1)
        self.conv7 = nn.Conv2d(192, 192, 3)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.class_conv = nn.Conv2d(192, n_classes, 1)



    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        maxPool3_out = F.relu(self.maxPool3(conv2_out))
        maxPool3_out_drop = F.dropout(maxPool3_out, .5)
        conv4_out = F.relu(self.conv4(maxPool3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        maxPool6_out = F.relu(self.maxPool6(conv5_out))
        maxPool6_out_drop = F.dropout(maxPool6_out, .5)
        conv7_out = F.relu(self.conv7(maxPool6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
"""
convPoolC="""class AllConvNet(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1)
        self.maxPool3 = nn.MaxPool2d(3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1)
        self.maxPool6 = nn.MaxPool2d(3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv2(conv2_out))
        maxPool3_out = F.relu(self.maxPool3(conv3_out))
        maxPool3_out_drop = F.dropout(maxPool3_out, .5)
        conv4_out = F.relu(self.conv4(maxPool3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        maxPool6_out = F.relu(self.maxPool6(conv6_out))
        maxPool6_out_drop = F.dropout(maxPool6_out, .5)
        conv7_out = F.relu(self.conv7(maxPool6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
"""


models=[modelC,convPoolC,allCnnC]
modelNames=["modelC","convPoolC","allCnnC"]
learningRates=[0.01,0.25,0.05,0.10]
for i in range(len(models)):
	for lr in learningRates:
		name=modelNames[i]+"_"+(str)(lr)+".py"
		temp=""
		with open('base','r') as f:
			lines=f.readlines()
			for line in lines:
				if(line=="modelCode\n"):
					temp+=models[i]+"\n"
				elif(line=="optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=0.9,weight_decay=0.001)\n"):
					temp+="optimizer = optim.SGD(model.parameters(), lr="+(str)(lr)+", momentum=0.9,weight_decay=0.001)\n"
				else:
					temp+=line
		createFile="touch models/"+name
		os.system(createFile)
		text_file = open("models/"+name,"w")
		text_file.write(temp)
		text_file.close()








