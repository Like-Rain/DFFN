import torch
import torch.nn as nn
import torch.nn.functional as F
# from . import block as B
# from . import SwinT

import pdb
import numpy as np
import sys
sys.path.append("..")
# from option import args
# import utility
from option import args
from model import block as B

def make_model(args):
    model = HNCT(args)
    return model

class HNCT(nn.Module):
    def __init__(self, args, in_nc=3, nf=50, num_modules=4, num_heads=5, out_nc=3):
        super(HNCT, self).__init__()
        upscale = args.scale[0]
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        self.B1 = B.HBCT(in_channels=nf, depth=2, num_heads=num_heads)
        self.B2 = B.HBCT(in_channels=nf, depth=2, num_heads=num_heads)
        self.B3 = B.HBCT(in_channels=nf, depth=2, num_heads=num_heads)
        self.B4 = B.HBCT(in_channels=nf, depth=2, num_heads=num_heads)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

    def forward(self, input_lr):
        out_fea_lr = self.fea_conv(input_lr)
        out_B1 = self.B1(out_fea_lr)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea_lr
        output = self.upsampler(out_lr)
        return output

if __name__ == '__main__':
    from thop import profile
    from thop import clever_format
    from torchinfo import summary
    # from torchstat import stat
    # from torchsummary import summary

    device = torch.device('cuda')
    model = HNCT(args)
    input = torch.randn(16, 3, 64, 64)
    input = input.to(device)
    model = model.to(device)
    # summary(model,(3,64,64),batch_size=16,device="cuda")  #torchsummary
    summary(model,(16,3,64,64),device="cuda")  #torchinfo

    print('parameters_count:',sum(p.numel() for p in model.parameters() if p.requires_grad))

    # # stat(model,(3,64,64))

    # flops, params = profile(model, inputs=(input, ))
    # print(flops, params)
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)

