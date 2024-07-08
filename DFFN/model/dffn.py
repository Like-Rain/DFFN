import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from functools import reduce
# from . import block as B
# from . import SwinT
import pdb
import sys
sys.path.append("..")
# from option import args
# import utility
from model import block as B
from option import args

import matplotlib.pyplot as plt

def make_model(args):
    model = DFFN(args)
    return model

def Gaussian_filter(img):
        c, h, w = img.shape[1:]
        #生成低通和高通滤波器
        lpf = torch.zeros((h,w))
        R = (h+w)//8  #或其他
        for x in range(w):
            for y in range(h):
                if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) < (R**2):
                    lpf[y,x] = 1
        hpf = 1-lpf
        hpf, lpf = hpf.cuda(), lpf.cuda()

        # X = transforms.unnormalize(X)  #注意fft只能处理非负数，通常X是标准化到正态分布的，这里需要把X再变换到[0,1]区间，unnormalize = lambda x: x*std + mu
        f = fft.fftn(img, dim=(2,3))
        # f = torch.roll(f,(h//2,w//2),dims=(2,3)) #移频操作,把低频放到中央
        f = torch.fft.fftshift(f)
        f_l = f * lpf
        f_h = f * hpf
        X_l = torch.abs(fft.ifftn(f_l,dim=(2,3)))
        X_h = torch.abs(fft.ifftn(f_h,dim=(2,3)))
        return X_h, X_l

class DFFN(nn.Module):
    def __init__(self, args, in_nc=3, nf=50, num_heads=5, out_nc=3, act_type='prelu'):
        super(DFFN, self).__init__()
        self.denoising = args.denoising
        if self.denoising:
            in_nc = args.n_colors
            self.conv_last = nn.Conv2d(nf, in_nc, 3, 1, 1)
        upscale = args.scale[0]
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        self.B1 = B.HBCT(in_channels=nf, depth=1, num_heads=num_heads)
        self.B2 = B.HBCT(in_channels=nf, depth=1, num_heads=num_heads)
        self.B3 = B.HBCT(in_channels=nf, depth=3, num_heads=num_heads)
        self.B4 = B.HBCT(in_channels=nf, depth=1, num_heads=num_heads)
        self.SK1 = B.FIM(nf, nf//4, act_type)
        self.SK2 = B.FIM(nf, nf//4, act_type)
        self.SK3 = B.FIM(nf, nf//4, act_type)
        # self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.sk_c = nf
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        f = nf // 1
        self.mlp = nn.Sequential(
            nn.Conv2d(nf, f*8, 1, groups=10, bias=False),
            B.activation(act_type),
            B.ShuffleBlock(groups=10),
            nn.Conv2d(f*8, nf*8, 1, groups=10, bias=False),
        )
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

    def forward(self, input_lr):      
        out_fea_lr = self.fea_conv(input_lr)
    
        if args.gaussian:
            # pdb.set_trace()
            high_lr, low_lr = Gaussian_filter(out_fea_lr)
        else:
            down_fea_lr = self.down(out_fea_lr)
            low_lr = F.interpolate(down_fea_lr, size = input_lr.size()[-2:], mode='bicubic', align_corners=True) 
            high_lr = out_fea_lr - low_lr

        out_A1 = self.B1(low_lr)
        out_B1 = self.B1(high_lr)

        out_A1_sk, out_B1_sk = self.SK1(out_A1, out_B1)

        out_A2 = self.B2(out_A1_sk)
        out_B2 = self.B2(out_B1_sk)

        out_A2_sk, out_B2_sk = self.SK2(out_A2, out_B2)
        out_A3 = self.B3(out_A2_sk)
        out_B3 = self.B3(out_B2_sk)

        out_A3_sk, out_B3_sk = self.SK3(out_A3, out_B3)
        out_A4 = self.B4(out_A3_sk)
        out_B4 = self.B4(out_B3_sk)
        
        output = []
        batch_size=input_lr.size(0)
        output.append(out_A1)
        output.append(out_A2)
        output.append(out_A3)
        output.append(out_A4)
        output.append(out_B1)
        output.append(out_B2)
        output.append(out_B3)
        output.append(out_B4)
        U=reduce(lambda x,y:x+y,output)      
        a_b = self.mlp(self.max_pool(U)+self.avg_pool(U))
        a_b=a_b.reshape(batch_size,8,self.sk_c,-1) 
        # pdb.set_trace()
        a_b=nn.Softmax(dim=1)(a_b)  
        a_b=list(a_b.chunk(8,dim=1))
        a_b=list(map(lambda x:x.reshape(batch_size,self.sk_c,1,1),a_b))
        V=list(map(lambda x,y:x*y,output,a_b))
        out_lr=reduce(lambda x,y:x+y,V)

        # out_B = self.c(torch.cat([out_A1, out_A2, out_A3, out_A4, out_B1, out_B2, out_B3, out_B4], dim=1))
        out_c = self.LR_conv(out_lr)
    
        out_lr = out_fea_lr + out_c

        if self.denoising:
            sr = self.conv_last(out_lr)
        else:
            sr = self.upsampler(out_lr)

        return sr
