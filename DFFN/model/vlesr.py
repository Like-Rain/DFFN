import torch
import torch.nn as nn
import sys
sys.path.append("..")
from model import common
from option import args

def make_model(args, parent=False):
    return VLESR(args)


def channel_shuffle(x, groups):
    B, C, H, W = x.data.size()
    channels_per_group = C // groups
    x = x.view(B, groups, channels_per_group, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(B, -1, H, W)
    return x


def Split_group(x, groups):
    B, C, H, W = x.data.size()
    group_channels = C // groups

    group_split = []
    for i in range(groups):
        group_split.append(x[:, i * group_channels:(i + 1) * group_channels, :, :])
    return group_split


class HGConv(nn.Module):

    def __init__(self, inChannels, group):
        super(HGConv, self).__init__()
        self.group = group
        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.group1_1x1conv = wn(nn.Conv2d(inChannels // group, inChannels // group, kernel_size=1))
        self.group2_1x1conv = wn(nn.Conv2d(inChannels // group, inChannels // group, kernel_size=1))
        self.group3_1x1conv = wn(nn.Conv2d(inChannels // group, inChannels // group, kernel_size=1))
        self.group4_1x1conv = wn(nn.Conv2d(inChannels // group, inChannels // group, kernel_size=1))

    def forward(self, x):
        groups = Split_group(x, self.group)

        group1 = self.group1_1x1conv(groups[0])
        group2 = self.group2_1x1conv(group1 + groups[1])
        group3 = self.group3_1x1conv(group2 + groups[2])
        group4 = self.group4_1x1conv(group3 + groups[3])

        out = torch.cat([group1, group2, group3, group4], dim=1)

        return out


# 替换3x3卷积层
class CDWConv(nn.Module):

    def __init__(self, inChannels, groups):
        super(CDWConv, self).__init__()
        self.groups = groups

        self.hgc_conv = HGConv(inChannels, self.groups)

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.dwconv = wn(nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1, stride=1, groups=inChannels))
        self.point_conv = wn(nn.Conv2d(inChannels, inChannels, kernel_size=1))

    def forward(self, x):
        # 需要搅拌，不然下次进到模块内分组不会改变
        hgc_conv = self.hgc_conv(x)
        shuffle = channel_shuffle(hgc_conv, self.groups)
        out = self.point_conv(self.dwconv(shuffle))

        return out


# 注意力
class HCALayer(nn.Module):

    def __init__(self, inChannels, reduction=16):
        super(HCALayer, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_mean = nn.Sequential(
            wn(nn.Conv2d(inChannels, inChannels // reduction, 1, padding=0, bias=True)),
            nn.ReLU(inplace=True),
            wn(nn.Conv2d(inChannels // reduction, inChannels, 1, padding=0, bias=True)),
            nn.Sigmoid()
        )

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_max = nn.Sequential(
            wn(nn.Conv2d(inChannels, inChannels // reduction, 1, padding=0, bias=True)),
            nn.ReLU(inplace=True),
            wn(nn.Conv2d(inChannels // reduction, inChannels, 1, padding=0, bias=True)),
            nn.Sigmoid()
        )

        self.local = nn.Sequential(
            wn(nn.Conv2d(inChannels, inChannels // reduction, 1, padding=0, bias=True)),
            nn.ReLU(inplace=True),
            wn(nn.Conv2d(inChannels // reduction, inChannels, 1, padding=0, bias=True)),
            nn.Sigmoid()
        )

    def forward(self, x):
        hca_mean = self.avg_pool(x)
        up_mean = self.conv_mean(hca_mean)

        hca_max = self.max_pool(x)
        bt_max = self.conv_max(hca_max)

        down_local = self.local(x)

        weight = up_mean + bt_max + down_local

        hca = x * weight
        return hca


# 残差
class RCB(nn.Module):

    def __init__(self, inChannels, groups=4):
        super(RCB, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        block = []
        block.append(CDWConv(inChannels, groups))
        block.append(nn.ReLU(inplace=True))
        block.append(CDWConv(inChannels, groups))
        block.append(nn.ReLU(inplace=True))
        self.conv_block = nn.Sequential(*block)
        self.compress = wn(nn.Conv2d(2 * inChannels, inChannels, kernel_size=1, padding=0, bias=True))

    def forward(self, x):
        conv_block = self.conv_block(x)
        conv_add = x + conv_block
        out = self.compress(torch.cat([x, conv_add], dim=1))

        return out


class RAB(nn.Module):

    def __init__(self, inChannels, groups=4):
        super(RAB, self).__init__()

        self.rcb0 = RCB(inChannels, groups)
        self.rcb1 = RCB(inChannels, groups)
        self.rcb2 = RCB(inChannels, groups)

        self.hca = HCALayer(inChannels)

    def forward(self, x):
        rcb0 = self.rcb0(x)
        rcb1 = self.rcb1(rcb0)
        rcb2 = self.rcb2(rcb1)

        hca = self.hca(rcb2)
        out = x + hca

        return out


# 特征融合
class FeatureFusion(nn.Module):

    def __init__(self, inChannels, kSize=1):
        super(FeatureFusion, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.Conv1 = nn.Sequential(
            wn(nn.Conv2d(inChannels, inChannels // 2, kSize, padding=0, bias=True)),
            nn.ReLU())
        self.Conv2 = nn.Sequential(
            wn(nn.Conv2d(inChannels, inChannels // 2, kSize, padding=0, bias=True)),
            nn.ReLU())
        self.fuse12 = nn.Sequential(
            wn(nn.Conv2d(inChannels, inChannels // 2, kSize, padding=0, bias=True)),
            nn.ReLU())

        self.Conv0 = nn.Sequential(
            wn(nn.Conv2d(inChannels, inChannels // 2, kSize, padding=0, bias=True)),
            nn.ReLU())
        self.Conv3 = nn.Sequential(
            wn(nn.Conv2d(inChannels, inChannels // 2, kSize, padding=0, bias=True)),
            nn.ReLU())
        self.fuse03 = nn.Sequential(
            wn(nn.Conv2d(inChannels * 3 // 2, inChannels, kSize, padding=0, bias=True)),
            nn.ReLU())

        self.hca = HCALayer(inChannels)

    def forward(self, x):
        Conv1 = self.Conv1(x[1])
        Conv2 = self.Conv2(x[2])
        fuse12 = self.fuse12(torch.cat([Conv1, Conv2], dim=1))

        Conv0 = self.Conv0(x[0])
        Conv3 = self.Conv3(x[3])
        fuse03 = self.fuse03(torch.cat([fuse12, Conv0, Conv3], dim=1))

        out = self.hca(fuse03)

        return out


class VLESR(nn.Module):

    def __init__(self, args):
        super(VLESR, self).__init__()

        scale = args.scale[0]
        nChannel = 3
        inChannels = 64
        # nDiff = 16
        groups = 4

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        wn = lambda x: torch.nn.utils.weight_norm(x)
        # 浅特征提取
        self.head = wn(nn.Conv2d(nChannel, inChannels, kernel_size=3, padding=1, stride=1))

        # 网络主干组成部分
        self.body_unit1 = RAB(inChannels, groups)
        self.body_unit2 = RAB(inChannels, groups)
        self.body_unit3 = RAB(inChannels, groups)
        self.body_unit4 = RAB(inChannels, groups)

        self.ff = FeatureFusion(inChannels)

        # 上采样
        modules_tail = [wn(nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1, bias=True)),
                        wn(nn.Conv2d(inChannels, 3 * (scale ** 2), kernel_size=3, padding=1, bias=True)),
                        nn.PixelShuffle(scale)]
        self.tail = nn.Sequential(*modules_tail)

        self.skip = nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        x = self.sub_mean(x)

        x1 = self.head(x)

        res1 = self.body_unit1(x1)
        res2 = self.body_unit2(res1)
        res3 = self.body_unit3(res2)
        res4 = self.body_unit4(res3)

        res = [res1, res2, res3, res4]
        ff = self.ff(res)

        body = ff + x1
        out = self.tail(body)
        skip = self.skip(x)
        out += skip

        x = self.add_mean(out)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

if __name__ == '__main__':
    from thop import profile
    from thop import clever_format
    from torchinfo import summary
    # from torchstat import stat
    # from torchsummary import summary

    device = torch.device('cuda')
    model = VLESR(args)
    input = torch.randn(16, 3, 48, 48)
    input = input.to(device)
    model = model.to(device)
    # summary(model,(3,64,64),batch_size=16,device="cuda")  #torchsummary
    summary(model, (16, 3, 48, 48), device="cuda")  #torchinfo

    # print('parameters_count:',sum(p.numel() for p in model.parameters() if p.requires_grad))

    # stat(model,(3,64,64))

    flops, params = profile(model, inputs=(input, ))
    print(flops, params)
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)