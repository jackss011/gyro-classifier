import torch.nn as nn
import torchvision.transforms as transforms
import math
from .binarized_modules import BinarizeLinear, BinarizeConv2d

__all__ = ['resnet_bin_relu']


def Binaryconv3x3(in_planes, out_planes, stride=1):
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def init_model(model):
    for m in model.modules():
        if isinstance(m, BinarizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None): # without do_bntan
        super(BasicBlock, self).__init__()
        self.conv1 = Binaryconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Binaryconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.bn2(out)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = BinarizeConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BinarizeConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(BinarizeConv2d(self.inplanes, planes * block.expansion,
                                                      kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10, block=BasicBlock, depth=18):
        super(ResNet_cifar10, self).__init__()
        self.inflate = 5
        self.inplanes = 16*self.inflate
        n = int((depth - 2) / 6)
        self.conv1 = BinarizeConv2d(3, 16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16*self.inflate)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16*self.inflate, n)
        self.layer2 = self._make_layer(block, 32*self.inflate, n, stride=2)
        self.layer3 = self._make_layer(block, 64*self.inflate, n, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.fc = BinarizeLinear(64*self.inflate, num_classes)

        init_model(self)
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 5*1e-2,
                'weight_decay': 1e-4, 'momentum': 0.9},
            81: {'lr': 1e-2},
            122: {'lr': 1e-3, 'weight_decay': 0},
            164: {'lr': 1e-4},
            200: {'lr': 1e-5},
            300: {'lr': 1e-6}
        }


def resnet_bin_relu(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    if dataset == 'cifar10':
        return ResNet_cifar10(num_classes=10, block=BasicBlock, depth=18)
