import torch
from torch import nn
from torch.nn import functional as F



class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding, groups=1, act=True):
        super().__init__()
        self.conv=nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn=nn.BatchNorm2d(out_c)
        # swapping from leakyrelu bc the extra param is worth it
        # if act is false no activation
        self.act=nn.PReLU(out_c) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DepthWiseBlock(nn.Module):
    def __init__(self, in_c, out_c, stride, expand, residual=False):
        super().__init__() 
        mid_c = in_c * expand
        self.use_residual = residual and (stride == 1) and (in_c == out_c)
        # pointwise conv to expand and mix channels
        self.pw = ConvBlock(in_c, mid_c, kernel=1, stride=1, padding=0, act=True)
        # depthwise conv for spatial feature extraction (groups is set to out_c so there is no channel mixing)
        self.dw = ConvBlock(mid_c, mid_c, kernel=3, stride=stride, padding=1, groups=mid_c, act=True)
        # pointwise linear to shrink channels back down
        self.pwl = ConvBlock(mid_c, out_c, kernel=1, stride=1, padding=0, act=False)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        out = self.pwl(out)
        if self.use_residual:
            out = out + x
        return out

class ResidualStack(nn.Module):
    def __init__(self, c, n):
        super().__init__()
        # n depthwise blocks with residual connections
        self.blocks = nn.Sequential(*[DepthWiseBlock(c, c, stride=1, expand=2, residual=True) for _ in range(n)])

    def forward(self, x):
        return self.blocks(x)

class MiniCNN(nn.Module):
    # adjust width multiplier to lighten the model (lower number = lighter)
    # lwk wouldn't put it below 0.75 unless you add more blocks
    def __init__(self, emb_dim=512, width_mult=0.75):
        super().__init__()
        # it's supposed to be faster if channels are divisible by 8 bc of byte alotment?
        # https://www.reddit.com/r/datascience/comments/119ak21/why_numbers_divisible_by_8_works_great_in_neural/
        def c(ch):
            ch = int(ch * width_mult)
            return max(8, int(round(ch / 8.0) * 8))

        c32 = c(32)
        c64 = c(64)
        c96 = c(96)
        c128 = c(128)
        c256 = c(256)

        # mixes channels and shrinks spatial map
        self.stem = ConvBlock(3, c64, kernel=3, stride=2, padding=1, act=True) # 56x56
        self.dw_stem = ConvBlock(c64, c64, kernel=3, stride=1, padding=1, groups=c64, act=True)
        
        self.stage2_down = DepthWiseBlock(c64, c64, stride=2, expand=2, residual=False) # 28x28
        self.stage2_res = ResidualStack(c64, n=2) # mobilefacnet has n = 4

        self.stage3_down = DepthWiseBlock(c64, c96, stride=2, expand=2, residual=False) # 14x14
        self.stage3_res = ResidualStack(c96, n=3) # mobilefacnet has n = 6

        self.stage4_down = DepthWiseBlock(c96, c128, stride=2, expand=2, residual=False) #7x7
        self.stage4_res = ResidualStack(c128, n=2) # mobilefacnet has n = 4

        self.conv_sep = ConvBlock(c128, c256, kernel=1, stride=1, padding=0, act=True)
        self.conv_dw = ConvBlock(c256, c256, kernel=7, stride=1, padding=0, groups=c256, act=False) # 1x1z

        self.fc = nn.Linear(c256, emb_dim, bias=False)
        self.bn = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.dw_stem(x)

        x = self.stage2_down(x)
        x = self.stage2_res(x)

        x = self.stage3_down(x)
        x = self.stage3_res(x)

        x = self.stage4_down(x)
        x = self.stage4_res(x)

        x = self.conv_sep(x)
        x = self.conv_dw(x)
        x = x.flatten(1)

        x = self.fc(x)
        x = self.bn(x)
        # normalizes to a unit hypersphere
        x = F.normalize(x, p=2, dim=1)
        return x