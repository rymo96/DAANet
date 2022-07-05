import torch.nn as nn
import torch
import math


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


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def group_conv_1x1_bn(inp, oup, expand_ratio):
    hidden_dim = oup // expand_ratio
    return nn.Sequential(
        nn.Conv2d(inp, hidden_dim, 1, 1, 0, groups=hidden_dim, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class SGBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False, att=None):
        super(SGBlock, self).__init__()
        assert stride in [1, 2]

        if isinstance(att, str):
            if "se" in att:
                att_block = SE(inp, int(att.split("_")[1]))
            elif "cbam" in att:
                att_block = CBAM(inp, int(att.split("_")[1]))
            elif "ca" in att:
                att_block = CA(inp, int(att.split("_")[1]))
            elif "daa" in att:
                att_block = DAA(inp, int(att.split("_")[1]), int(att.split("_")[2]), int(att.split("_")[3]), int(att.split("_")[4]), int(att.split("_")[5]))
        else:
            att_block = nn.Sequential()

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)# + 16

        #self.relu = nn.ReLU6(inplace=True)
        self.identity = False
        self.identity_div = 1
        self.expand_ratio = expand_ratio
        if expand_ratio == 2:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                att_block,
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif inp != oup and stride == 1 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        elif inp != oup and stride == 2 and keep_3x3==False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            if keep_3x3 == False:
                self.identity = True
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                att_block,
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, 1, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)

        if self.identity:
            shape = x.shape
            id_tensor = x[:,:shape[1]//self.identity_div,:,:]
            # id_tensor = torch.cat([x[:,:shape[1]//self.identity_div,:,:],torch.zeros(shape)[:,shape[1]//self.identity_div:,:,:].cuda()],dim=1)
            # import pdb; pdb.set_trace()
            out[:,:shape[1]//self.identity_div,:,:] = out[:,:shape[1]//self.identity_div,:,:] + id_tensor
            return out #+ x
        else:
            return out

class MobileNeXt(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., att=None):
        super(MobileNeXt, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [2,  96, 1, 2],
            [6, 144, 1, 1],
            [6, 192, 3, 2],
            [6, 288, 3, 2],
            [6, 384, 4, 1],
            [6, 576, 4, 2],
            [6, 960, 3, 1],
            [6,1280, 1, 1],
        ]
        #self.cfgs = [
        #    # t, c, n, s
        #    [1,  16, 1, 1],
        #    [4,  24, 2, 2],
        #    [4,  32, 3, 2],
        #    [4,  64, 3, 2],
        #    [4,  96, 4, 1],
        #    [4, 160, 3, 2],
        #    [4, 320, 1, 1],
        #]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = SGBlock
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            if c == 1280 and width_mult < 1:
                output_channel = 1280
            layers.append(block(input_channel, output_channel, s, t, n==1 and s==1, att=att))
            input_channel = output_channel
            for i in range(n-1):
                layers.append(block(input_channel, output_channel, 1, t, att=att))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        input_channel = output_channel
        output_channel = _make_divisible(input_channel, 4) # if width_mult == 0.1 else 8) if width_mult > 1.0 else input_channel
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
                # nn.Dropout(0.2),
                nn.Linear(output_channel, num_classes)
                )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)