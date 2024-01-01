import os 
import torch
import torch.nn as nn
from matplotlib import puplot as plt
from torch import optim 

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=256,device="cuda"):
        super().__init__()
        self.device=device
        self.time_dim=time_dim
        self.inc=DoubleConv(in_channels,64)
        self.down1=Down(64,128)
        self.sa1=SelfAttention(128,32)
        self.down2=Down(128,256)
        self.sa2=SelfAttention(256,16)
        self.down3=Down(256,256)
        self.sa3=SelfAttention(256,8)

        



