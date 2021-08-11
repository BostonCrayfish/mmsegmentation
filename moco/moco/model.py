import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

def Conv_block(in_channels, out_channels, k, s=1, p=0, d=1, bias=False):
    return nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(k, k),
                    stride=(s, s),
                    padding=(p, p),
                    dilation=(d, d),
                    bias=bias)),
                ('bn', nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                ('activate', nn.ReLU(inplace=True))]))

class ASPPModule(nn.ModuleList):
    def __init__(self, dilations, in_channels, channels):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        for dilation in dilations:
            self.append(
                Conv_block(self.in_channels,
                           self.channels,
                           1 if dilation == 1 else 3,
                           d=dilation,
                           p=0 if dilation == 1 else dilation))
    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))
        return aspp_outs

# setting for aspp stride 16
# dilations=(1, 1, 1, 2)
# strides=(1, 2, 2, 1)

class Decode_head(nn.Module):
    def __init__(self):
        super(Decode_head, self).__init__()
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv_block(2048, 512, 1))
        self.aspp_modules = ASPPModule(dilations=(1, 6, 12, 18), in_channels=2048, channels=512)
        self.bottleneck = Conv_block(2560, 512, 3, p=1)
        self.contrast_conv = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 128, 1))
    def forward(self, x):
        aspp_outs = [
            nn.functional.interpolate(self.image_pool(x), x.size()[2:], mode='bilinear')]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        output = self.contrast_conv(output)
        return output

class Encoder_Decoder(nn.Module):
    def __init__(self):
        super(Encoder_Decoder, self).__init__()
        base_backbone = models.__dict__['resnet50']
        model_base = base_backbone()
        self.backbone = nn.Sequential(OrderedDict([('conv1', model_base.conv1),
                                                    ('bn1', model_base.bn1),
                                                    ('relu', model_base.relu),
                                                    ('maxpool', model_base.maxpool),
                                                    ('layer1', model_base.layer1),
                                                    ('layer2', model_base.layer2),
                                                    ('layer3', model_base.layer3),
                                                    ('layer4', model_base.layer4)]))
        del model_base
        self.backbone.layer4[0].downsample[0] = nn.Conv2d(
            1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.layer4[0].conv2 = nn.Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.layer4[1].conv2 = nn.Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.backbone.layer4[2].conv2 = nn.Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.decode_head = Decode_head()
    def forward(self, img):
        x = self.backbone(img)
        return self.decode_head(x)
