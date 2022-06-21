import torch
import torch.nn as nn
        


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


class SSE(nn.Module):
    def __init__(self, planes, groups=1):
        super(SSE, self).__init__()
        self.context = nn.AdaptiveAvgPool2d(1)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=1, groups=groups),
            # nn.BatchNorm2d(planes),
        )
        
    def forward(self, x):
        out = self.context(x)
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


class XCA(nn.Module):
    def __init__(self, planes, ratio=16):
        super(XCA, self).__init__()
        mip = max(8, planes // ratio)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=mip, kernel_size=1),
            nn.BatchNorm2d(mip),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mip, out_channels=planes, kernel_size=1),
        )
        
    def forward(self, x):
        out = torch.unsqueeze(torch.mean(x, dim=3), dim=3)
        out = self.fusion(out)

        return x * out.sigmoid()


class YCA(nn.Module):
    def __init__(self, planes, ratio=16):
        super(YCA, self).__init__()
        mip = max(8, planes // ratio)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=mip, kernel_size=1),
            nn.BatchNorm2d(mip),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mip, out_channels=planes, kernel_size=1),
        )
        
    def forward(self, x):
        out = torch.unsqueeze(torch.mean(x, dim=2), dim=3)
        out = self.fusion(out)
        out = torch.transpose(out, dim0=2, dim1=3)

        return x * out.sigmoid()


class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2)
        
        def forward(self, x):
            # x: input features with shape [b, c, h, w]
            b, c, h, w = x.size()
            
            # feature descriptor on the global spatial information
            out = self.avg_pool(x)
            
            # Two different branches of ECA module
            out = self.fusion(out.squeeze(-1).transpose(-1, -2))
            out = out.transpose(-1, -2).unsqueeze(-1)
            
            return x * out.sigmoid()


class CSA(nn.Module):
    def __init__(self, planes, k_size=3):
        super(CSA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2)
        self.bn = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2)

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        out = self.avg_pool(x)

        # Two different branches of ECA module
        out = self.conv1(out.squeeze(-1).transpose(-1, -2))
        out = out.transpose(-1, -2).unsqueeze(-1)
        out = self.bn(out)
        out = self.conv2(out.squeeze(-1).transpose(-1, -2))
        out = out.transpose(-1, -2).unsqueeze(-1)

        return x * out.sigmoid()


class DA(nn.Module):
    def __init__(self, planes, k_size=7, ratio=4, use_c=True, use_x=True, use_y=True):
        super(DA, self).__init__()
        mip = max(8, planes // ratio)
        self.use_c = (use_c == 1)
        self.use_x = (use_x == 1)
        self.use_y = (use_y == 1)

        if self.use_c:
            self.fusion_c = nn.Sequential(
                nn.Conv2d(in_channels=planes, out_channels=mip, kernel_size=1),
                # nn.BatchNorm2d(mip),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=mip, out_channels=planes, kernel_size=1),
                nn.Sigmoid(),
            )

        if self.use_x:
            self.fusion_x = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=(k_size, 1), padding=(k_size//2, 0)),
                # nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 1, kernel_size=(k_size, 1), padding=(k_size//2, 0)),
                nn.Sigmoid()
            )

        if self.use_y:
            self.fusion_y = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=(1, k_size), padding=(0, k_size//2)),
                # nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 1, kernel_size=(1, k_size), padding=(0, k_size//2)),
                nn.Sigmoid()
            )

    def forward(self, input):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = input.size()
        output = input

        # feature descriptor on the global spatial information
        if self.use_c:
            out_c = input.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            output = output * self.fusion_c(out_c)
        if self.use_x:
            out_x = input.mean(dim=1, keepdim=True).mean(dim=3, keepdim=True)
            output = output * self.fusion_x(out_x)
        if self.use_y:
            out_y = input.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
            output = output * self.fusion_y(out_y)

        return output


class BA(nn.Module):
    def __init__(self, planes, ratio=32):
        super(BA, self).__init__()
        mip = max(2, planes // ratio)
        self.context = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=mip, kernel_size=1),
            nn.BatchNorm2d(mip),
            nn.Conv2d(in_channels=mip, out_channels=mip, kernel_size=3, padding=1),
            nn.BatchNorm2d(mip),
            nn.Conv2d(in_channels=mip, out_channels=planes, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        out = self.context(x)
        return x * out.sigmoid()


class ShuffleV2BlockV2(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride, att=None):
        super(ShuffleV2BlockV2, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]

        if isinstance(att, str):
            if "sse" in att:
                att_block = SSE(outputs, int(att.split("_")[1]))
            elif "cbam" in att:
                att_block = CBAM(outputs, int(att.split("_")[1]))
            elif "ba" in att:
                att_block = BA(outputs, int(att.split("_")[1]))
            elif "se" in att:
                att_block = SE(outputs, int(att.split("_")[1]))
            elif "eca" in att:
                att_block = ECA(outputs, int(att.split("_")[1]))
            elif "xca" in att:
                att_block = XCA(outputs, int(att.split("_")[1]))
            elif "yca" in att:
                att_block = YCA(outputs, int(att.split("_")[1]))
            elif "ca" in att:
                att_block = CA(outputs, int(att.split("_")[1]))
            elif "csa" in att:
                att_block = CSA(outputs, int(att.split("_")[1]))
            elif "da" in att:
                att_block = DA(outputs, int(att.split("_")[1]), int(att.split("_")[2]), int(att.split("_")[3]), int(att.split("_")[4]), int(att.split("_")[5]))
        else:
            att_block = nn.Sequential()

        branch_main.append(att_block)
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]

            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = torch.chunk(old_x, chunks=2, dim=1)
            new_x = torch.cat((x_proj, self.branch_main(x)), 1)
            
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            new_x = torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

        return self.channel_shuffle(new_x)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize, num_channels // 2, 2, height * width)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batchsize, num_channels, height, width)
        return x


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000, base=176, layer=[4,8,4], kernel=3, drop=0.0, att=None):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = layer
        self.stage_out_channels = [base, base*2, base*4]

        # building first layer
        input_channel = 24
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleV2BlockV2(input_channel, output_channel, mid_channels=output_channel // 2, ksize=kernel, stride=2, att=att))
                else:
                    self.features.append(ShuffleV2BlockV2(input_channel // 2, output_channel, mid_channels=output_channel // 2, ksize=kernel, stride=1, att=att))

                input_channel = output_channel
                
        self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, 1024, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(1024, num_classes, bias=False)
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)

        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
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

if __name__ == "__main__":
    model = ShuffleNetV2()
    # print(model)

    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs.size())
