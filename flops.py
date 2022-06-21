import torch
from thop import profile, clever_format
from network.mobilenetv2 import MobileNetV2
from network.shufflenetv2 import ShuffleNetV2
from network.mobilenext import MobileNeXt
from network.mobilenetv3 import mobilenet_v3_small
from network.ghostnet import ghostnet

if __name__ == "__main__":
    input = torch.randn(1, 3, 224, 224)

    model = MobileNetV2(att="se_8")
    # model = MobileNetV2(att="cbam_24")
    # model = MobileNetV2(att="ca_32")
    # model = MobileNetV2(att="da_7_8_1_1_1")

    # model = MobileNeXt(att="se_8")
    # model = MobileNeXt(att="cbam_24")
    # model = MobileNeXt(att="ca_32")
    # model = MobileNeXt(att="da_7_8_1_1_1")

    # model = ShuffleNetV2(att="se_8")
    # model = ShuffleNetV2(att="cbam_24")
    # model = ShuffleNetV2(att="ca_32")
    # model = ShuffleNetV2(att="da_7_8_1_1_1")

    # model = mobilenet_v3_small(att="se_4")
    # model = mobilenet_v3_small(att="cbam_24")
    # model = mobilenet_v3_small(att="ca_32")
    # model = mobilenet_v3_small(att="da_7_4_1_1_1")

    # model = ghostnet(width=1.0, att="se_4")
    # model = ghostnet(width=1.0, att="cbam_24")
    # model = ghostnet(width=1.0, att="ca_32")
    # model = ghostnet(width=1.0, att="da_7_4_1_1_1")

    # print(model)
    model.eval()
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")

    print('Flops:  ', macs)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Params: {:.3f}M'.format(num_params/1e6))
