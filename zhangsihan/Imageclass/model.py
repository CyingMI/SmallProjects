import torch
import torch.nn as nn

class ResBlock(nn.Module):

    expansion=4

    def __init__(self,in_channels,hid_channels,stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels,hid_channels,kernel_size=1,padding=0),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(),
            nn.Conv2d(hid_channels,hid_channels,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(),
            nn.Conv2d(hid_channels,self.expansion*hid_channels,kernel_size=1,padding=0),
            nn.BatchNorm2d(self.expansion*hid_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*hid_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*hid_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*hid_channels))
        
    def forward(self,x):
        return torch.relu(self.shortcut(x) + self.cnn(x))
    
class ResNet(nn.Module):
    def __init__(self,num_res=20):
        super().__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage1=nn.Sequential(
            ResBlock(64, 64, stride=1),  
            ResBlock(256, 64, stride=1),
            ResBlock(256, 64, stride=1)
        )
        
        self.stage2=nn.Sequential(
            ResBlock(256, 128, stride=2),
            ResBlock(512, 128, stride=1),
            ResBlock(512, 128, stride=1),
            ResBlock(512, 128, stride=1)
        )

        self.stage3=nn.Sequential(
            ResBlock(512, 256, stride=2),
            ResBlock(1024, 256, stride=1),
            ResBlock(1024, 256, stride=1),
            ResBlock(1024, 256, stride=1),
            ResBlock(1024, 256, stride=1),
            ResBlock(1024, 256, stride=1)
        )

        self.stage4=nn.Sequential(
            ResBlock(1024, 512, stride=2),
            ResBlock(2048, 512, stride=1),
            ResBlock(2048, 512, stride=1)
        )

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))

        self.fc=nn.Linear(512*ResBlock.expansion,num_res)

    def forward(self,x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    


        


