import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self,in_channels,hidden_channels_1,hidden_channels_2,hidden_channels_3,hidden_channels_4,out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels_1 = hidden_channels_1
        self.hidden_channels_2 = hidden_channels_2
        self.hidden_channels_3 = hidden_channels_3
        self.hidden_channels_4 = hidden_channels_4
        self.out_channels = out_channels
        self.fc=nn.Sequential(
            nn.Linear(in_channels,hidden_channels_1),
            nn.Tanh(),
            nn.Linear(hidden_channels_1,hidden_channels_2),
            nn.Tanh(),
            nn.Linear(hidden_channels_2,hidden_channels_3),
            nn.Tanh(),
            nn.Linear(hidden_channels_3,hidden_channels_4),
            nn.Tanh(),
            nn.Linear(hidden_channels_4,out_channels)
        )
    def forward(self,x,t):
        inp=torch.cat((x,t),dim=1)
        return self.fc(inp) 