import torch.nn as nn
import torch
import numpy as np
import math
from .binarized_modules import TernarizeLinear, TernarizeConv2d, Ternarize

__all__ = ['resnet_ternary'] 


def Ternaryconv3x3(in_planes, out_planes, delta=0.3, stride=1):
    return TernarizeConv2d(in_planes, out_planes, delta=delta, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def init_model(model):  
    for m in model.modules():  
        if isinstance(m, TernarizeConv2d):  
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module): 
    expansion = 1 

    def __init__(self, inplanes, planes, delta=0.3, stride=1, downsample=None, do_bntan=True):
        super(BasicBlock, self).__init__()

        self.conv1 = Ternaryconv3x3(inplanes, planes, delta, stride) 
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)

        self.conv2 = Ternaryconv3x3(planes, planes, delta)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.delta = delta
        self.downsample = downsample
        self.do_bntan = do_bntan  
        self.stride = stride

    def forward(self, x: torch.Tensor):
        residual = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)
        out = self.conv2(out)
        if self.downsample is not None:
            if residual.data.max() > 1:  
                import pdb;
                pdb.set_trace()
            residual = self.downsample(residual) 
        out += residual 
        if self.do_bntan:
            out = self.bn2(out)
            out = self.tanh2(out) 
        return out


class Bottleneck(nn.Module):  
    expansion = 4 

    def __init__(self, inplanes, planes, delta=0.3, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = TernarizeConv2d(inplanes, planes, delta=delta, kernel_size=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(planes) 

        self.conv2 = TernarizeConv2d(planes, planes, delta=delta, kernel_size=3, stride=stride, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = TernarizeConv2d(planes, planes * 4, delta=delta, kernel_size=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.tanh = nn.Hardtanh(inplace=True)  
        self.delta = delta
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x 
        import pdb;
        pdb.set_trace()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.tanh(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)  
        out += residual 
        if self.do_bntan:  
            out = self.bn2(out)  
            out = self.tanh2(out) 
        return out


class ResNet(nn.Module):  

    def __init__(self):
        super(ResNet, self).__init__()


    def _make_layer(self, block, planes, blocks, delta=0.3, stride=1, do_bntan=True):  
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                TernarizeConv2d(self.inplanes,  planes * block.expansion, delta=delta, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(self.inplanes, planes, delta=delta, stride=stride, downsample=downsample)] 

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes, delta=delta))

        layers.append(block(self.inplanes, planes, delta=delta, do_bntan=do_bntan))  
        return nn.Sequential(*layers) 


    def forward(self, x):
        x = self.conv1(x)  
        x = self.maxpool(x) 
        x = self.bn1(x)
        x = self.tanh1(x)

        x = self.layer1(x) 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1) 
        x = self.bn2(x)
        x = self.tanh2(x)

        x = self.fc(x)
        x = self.bn3(x)

        x = self.logsoftmax(x)
        return x


    def count_not_ternary(self):
        count = 0
        for name, param in self.named_parameters():
            if param is not None:
                for p in param.view(-1):
                    if p.item() != 0 and p.item() != 1 and p.item() != -1:
                        #print(f'p: {p.item()}, Layer Name: {name}, Parameter Shape: {param.shape}')
                        count += 1
            else:
                print(f"{param} is None")
        return count

    '''
    def count_zero_weights(self):
        zeros = 0
        for param in self.parameters():
            if param is not None:
                zeros += param.numel() - param.nonzero().size(0)
        return zeros
    '''

    def count_zero_weights(self):
        zeros = 0
        for param in self.parameters():
            if param is not None:
                zeros += torch.sum(torch.eq(param, 0)).item()
        return zeros
    
    def count_one_weights(self):
        ones_count = 0
        for param in self.parameters():
            if param is not None:
                ones_count += torch.sum(torch.eq(param, 1)).item()
        return ones_count

    def count_minus_one_weights(self):
        minus_ones_count = 0
        for param in self.parameters():
            if param is not None:
                minus_ones_count += torch.sum(torch.eq(param, -1)).item()
        return minus_ones_count
    
    def return_parameter_abs_avg(self):
        sum_abs = 0
        num_param = 0

        for param in self.parameters(): 
            if param is not None and hasattr(param, 'org'):
                num_param += param.numel()
                sum_abs += torch.sum(torch.abs(param.org)).item()
        abs_avg = sum_abs/num_param
        return abs_avg

    
    def return_weights(self):
        params = []
        for k, v in self.state_dict().items():  
            if 'weight' in k: 
                params.append(torch.flatten(v))  
        conc = params[0]
        for i in range(1, params.__len__()):
            conc = torch.concat([conc, params[i]])  
        # return params, np.array(conc)
        # return params
        return np.array(conc)

    def return_weights_org(self):
        params = []
        for p in self.parameters():
            if hasattr(p, 'org'):
                params.append(torch.flatten(p.org))
        conc = params[0]
        for i in range(1, params.__len__()):
            conc = torch.concat([conc, params[i]]) 
        # return params, np.array(conc)
        # return params
        return np.array(conc.cpu())


class ResNet_imagenet(ResNet):  
    def __init__(self, num_classes=1000, block=Bottleneck, layers=[3, 4, 23, 3], delta=0.3):

        super(ResNet_imagenet, self).__init__()
        self.delta = delta
        self.inplanes = 64
        
        self.conv1 = TernarizeConv2d(3, 64, delta=delta, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.tanh = nn.Hardtanh(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], delta=delta)  
        self.layer2 = self._make_layer(block, 128, layers[1], delta=delta, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], delta=delta, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], delta=delta, stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = TernarizeLinear(512 * block.expansion, num_classes, delta=delta)

        init_model(self)
        self.regime = {0: {'optimizer': 'SGD', 'lr': 1e-1, 'weight_decay': 1e-4, 'momentum': 0.9},
                       30: {'lr': 1e-2},
                       60: {'lr': 1e-3, 'weight_decay': 0},
                       90: {'lr': 1e-4}}


class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10, block=BasicBlock, depth=18, delta=0.3):
        super(ResNet_cifar10, self).__init__()
        self.delta = delta
        self.inflate = 1  
        self.inplanes = 16 * self.inflate
        n = int((depth - 2) / 6) 
        self.conv1 = TernarizeConv2d(3, 16 * self.inflate, delta=delta, kernel_size=3, stride=1, padding=1,
                                     bias=False) 
        self.maxpool = lambda x: x
        self.bn1 = nn.BatchNorm2d(16 * self.inflate)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block, 16 * self.inflate, n, delta=delta)
        self.layer2 = self._make_layer(block, 32 * self.inflate, n, delta=delta, stride=2)
        self.layer3 = self._make_layer(block, 64 * self.inflate, n, delta=delta, stride=2, do_bntan=False)
        self.layer4 = lambda x: x  
        self.avgpool = nn.AvgPool2d(8)
        self.bn2 = nn.BatchNorm1d(64 * self.inflate)
        self.bn3 = nn.BatchNorm1d(10)
        self.logsoftmax = nn.LogSoftmax()
        self.fc = TernarizeLinear(64 * self.inflate, num_classes, delta=delta)

        init_model(self)
        # self.regime = {0: {'optimizer': 'SGD', 'lr': 1e-1, 'weight_decay': 1e-4, 'momentum': 0.9},
        # 	       81: {'lr': 1e-4},
        #	       122: {'lr': 1e-5, 'weight_decay': 0},
        #	       164: {'lr': 1e-6}}
        
        self.regime = {0: {'optimizer': 'Adam', 'lr': 5e-3},
                       101: {'lr': 1e-3},
                       142: {'lr': 5e-4},
                       184: {'lr': 1e-4},
                       220: {'lr': 1e-5}
                       }
        '''
        self.regime = {0: {'optimizer': 'Adam', 'lr': 3e-3},
                       101: {'lr': 1e-3},
                       142: {'lr': 5e-4},
                       184: {'lr': 1e-4},
                       220: {'lr': 1e-5}}
        '''

def resnet_ternary(**kwargs):  
    num_classes, depth, dataset, delta = map(kwargs.get, ['num_classes', 'depth', 'dataset', 'delta'])

    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 50
        if depth == 18:
            return ResNet_imagenet(num_classes=num_classes, block=BasicBlock, layers=[2, 2, 2, 2], delta=delta)
        if depth == 34:
            return ResNet_imagenet(num_classes=num_classes, block=BasicBlock, layers=[3, 4, 6, 3], delta=delta)
        if depth == 50:
            return ResNet_imagenet(num_classes=num_classes, block=Bottleneck, layers=[3, 4, 6, 3], delta=delta)
        if depth == 101:
            return ResNet_imagenet(num_classes=num_classes, block=Bottleneck, layers=[3, 4, 23, 3], delta=delta)
        if depth == 152:
            return ResNet_imagenet(num_classes=num_classes, block=Bottleneck, layers=[3, 8, 36, 3], delta=delta)
    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 18
        return ResNet_cifar10(num_classes=num_classes, block=BasicBlock, depth=depth, delta=delta)
