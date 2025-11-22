# bifpn.py yang lebih komprehensif
import torch
import torch.nn as nn

class BiFPN_Block(nn.Module):
    def __init__(self, channels, epsilon=1e-4):
        super(BiFPN_Block, self).__init__()
        self.epsilon = epsilon
        
        # Feature fusion weights
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        
        # Convolution layers untuk feature refinement
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()
        
    def forward(self, p3, p4, p5):
        # Top-down path
        w1 = self.w1 / (torch.sum(self.w1, dim=0) + self.epsilon)
        w2 = self.w2 / (torch.sum(self.w2, dim=0) + self.epsilon)
        
        # P5 -> P4
        p5_up = nn.functional.interpolate(p5, size=p4.shape[2:], mode='nearest')
        p4_td = w1[0] * p4 + w1[1] * p5_up
        
        # P4 -> P3  
        p4_up = nn.functional.interpolate(p4_td, size=p3.shape[2:], mode='nearest')
        p3_out = w2[0] * p3 + w2[1] * p4_up + w2[2] * p5_up
        
        # Bottom-up path (opsional, untuk completeness)
        p3_dn = self.conv(p3_out)
        p3_dn = self.bn(p3_dn)
        p3_dn = self.act(p3_dn)
        
        return p3_out, p4_td, p5