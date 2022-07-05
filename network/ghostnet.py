"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.sigmoid = HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SE(nn.Module):
    def __init__(self, planes, ratio=16):
        super(SE, self).__init__()
        mip = max(8, planes // ratio)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=mip, kernel_size=1),
            # nn.BatchNorm2d(mip),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mip, out_channels=planes, kernel_size=1),
        )

    def forward(self, x):
        out = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        out = self.fusion(out)
        return x * out.sigmoid()


class CBAM(nn.Module):
    def __init__(self, planes, ratio=32, kernel_size=7):
        super(CBAM, self).__init__()
        # channel att
        mip = max(8, planes // ratio)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fusion1 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=mip, kernel_size=1),
            # nn.BatchNorm2d(mip),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mip, out_channels=planes, kernel_size=1),
        )

        # spatial att
        self.fusion2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        out1 = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True) + self.max_pool(x)
        out1 = self.fusion1(out1)
        out1 = out1.sigmoid()

        out2 = torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        out2 = self.fusion2(out2)
        out2 = out2.sigmoid()

        return x * out1 * out2


class CA(nn.Module):
    def __init__(self, planes, ratio=16):
        super(CA, self).__init__()
        mip = max(8, planes // ratio)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=mip, kernel_size=1),
            nn.BatchNorm2d(mip),
            HSwish()
        )
        self.sig_x = nn.Sequential(
            nn.Conv2d(in_channels=mip, out_channels=planes, kernel_size=1),
            nn.Sigmoid()
        )

        self.sig_y = nn.Sequential(
            nn.Conv2d(in_channels=mip, out_channels=planes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.size()
        out_x = torch.unsqueeze(torch.mean(x, dim=3), dim=3)
        out_y = torch.unsqueeze(torch.mean(x, dim=2), dim=3)

        out = torch.cat((out_x, out_y), dim=2)
        out = self.fusion(out)
        out_x, out_y = torch.split(out, [h, w], dim=2)
        out_y = torch.transpose(out_y, dim0=2, dim1=3)

        return x * self.sig_x(out_x) * self.sig_y(out_y)


class DAA(nn.Module):
    def __init__(self, planes, k_size=7, ratio=4, use_c=True, use_x=True, use_y=True):
        super(DAA, self).__init__()
        mip = max(8, planes // ratio)
        self.use_c = (use_c == 1)
        self.use_x = (use_x == 1)
        self.use_y = (use_y == 1)

        if self.use_x:
            self.fusion_x = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=(k_size, 1), padding=(k_size // 2, 0), bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 1, kernel_size=(k_size, 1), padding=(k_size // 2, 0)),
                nn.Sigmoid()
            )

        if self.use_y:
            self.fusion_y = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=(1, k_size), padding=(0, k_size // 2), bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 1, kernel_size=(1, k_size), padding=(0, k_size // 2)),
                nn.Sigmoid()
            )

        if self.use_c:
            self.fusion_c = nn.Sequential(
                nn.Conv2d(in_channels=planes, out_channels=mip, kernel_size=1, bias=False),
                nn.BatchNorm2d(mip),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=mip, out_channels=planes, kernel_size=1),
                nn.Sigmoid(),
            )

    def forward(self, input):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = input.size()
        output = input

        # feature descriptor on the global spatial information
        if self.use_x:
            out_x = input.mean(dim=1, keepdim=True).mean(dim=3, keepdim=True)
            output = output * self.fusion_x(out_x)
        if self.use_y:
            out_y = input.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
            output = output * self.fusion_y(out_y)
        if self.use_c:
            out_c = input.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            output = output * self.fusion_c(out_c)

        return output



class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, act_layer=nn.ReLU, se_ratio=0., att=None):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=dw_kernel_size//2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se and att is not None:
            if "se" in att:
                self.se = SE(mid_chs, int(att.split("_")[1]))
            elif "cbam" in att:
                self.se = CBAM(mid_chs, int(att.split("_")[1]))
            elif "ca" in att:
                self.se = CA(mid_chs, int(att.split("_")[1]))
            elif "daa" in att:
                self.se = DAA(mid_chs, int(att.split("_")[1]), int(att.split("_")[2]), int(att.split("_")[3]), int(att.split("_")[4]), int(att.split("_")[5]))
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            # self.shortcut = None
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride, padding=dw_kernel_size//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        if self.shortcut != None:
            x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2, att=None):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                              se_ratio=se_ratio, att=att))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        
        self.blocks = nn.Sequential(*stages)        

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

        self.init_params()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x).flatten(1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

    def init_params(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s 
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    return GhostNet(cfgs, **kwargs)


if __name__=='__main__':
    model = ghostnet(width=1.0)
    model.eval()
    x = torch.randn(1,3,224,224)
    y = model(x)
    torch.save(model.state_dict(), 'ghostnet.x1.0.pth')
    print(y.size()) 
