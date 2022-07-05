from torch import nn
from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
from torch.nn.modules.activation import Sigmoid
__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

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
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2),
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
        n,c,h,w = x.size()
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
                nn.Conv2d(1, 1, kernel_size=(k_size, 1), padding=(k_size//2, 0), bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 1, kernel_size=(k_size, 1), padding=(k_size//2, 0)),
                nn.Sigmoid()
            )

        if self.use_y:
            self.fusion_y = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=(1, k_size), padding=(0, k_size//2), bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 1, kernel_size=(1, k_size), padding=(0, k_size//2)),
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


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, kernel_size//2, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, att=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if isinstance(att, str):
            if "se" in att:
                att_block = SE(hidden_dim, int(att.split("_")[1]))
            elif "cbam" in att:
                att_block = CBAM(hidden_dim, int(att.split("_")[1]))
            elif "ca" in att:
                att_block = CA(hidden_dim, int(att.split("_")[1]))
            elif "daa" in att:
                att_block = DAA(hidden_dim, int(att.split("_")[1]), int(att.split("_")[2]), int(att.split("_")[3]), int(att.split("_")[4]), int(att.split("_")[5]))
        else:
            att_block = nn.Sequential()

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # att
            att_block,
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None,
                 att=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.conv_first = ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)

        # building inverted residual blocks
        features = []
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, att=att))
                input_channel = output_channel
        self.features = nn.Sequential(*features)

        self.conv_last = ConvBNReLU(input_channel, last_channel, kernel_size=1, norm_layer=norm_layer)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(last_channel)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.classifier = nn.Linear(last_channel, num_classes, bias=False)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)


    def _forward_impl(self, x):
        x = self.conv_first(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.gap(x).flatten(1)
        x = self.drop(self.relu(self.bn(x)))
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

