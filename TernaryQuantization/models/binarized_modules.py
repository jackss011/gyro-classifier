import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np


def Ternarize(tensor, delta):
    cond1 = torch.abs(tensor) < delta
    cond2 = tensor >= delta
    cond3 = tensor <= -delta
    t1 = torch.where(cond1, torch.tensor(0.).cuda(), tensor)
    t2 = torch.where(cond2, torch.tensor(1.).cuda(), t1)
    t3 = torch.where(cond3, torch.tensor(-1.).cuda(), t2)
    return t3



def Binarize(tensor, quant_mode='det'):  
    if quant_mode == 'det': 
        # return tensor.sign()
        return torch.where(tensor >= 0, 1.,
                           -1.)  
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(
            -1)  



def Quantize(tensor, quant_mode='det', params=None, numBits=2): 
    tensor.clamp_(-2 ** (numBits - 1), 2 ** (numBits - 1))
    if quant_mode == 'det': 
        tensor = tensor.mul(2 ** (numBits - 1)).round().div(2 ** (numBits - 1))
    else:  
        tensor = tensor.mul(2 ** (numBits - 1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2 ** (numBits - 1))
    return tensor


class TernarizeLinear(nn.Linear):  # TernarizeLinear
    def __init__(self, *kargs, delta=0.1, **kwargs):
        super(TernarizeLinear, self).__init__(*kargs, **kwargs)
        self.delta = delta

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        # if input.size(1) != 320: 
        input.data = Ternarize(input.data, self.delta) # ternary activations
        # self.weight.data = Ternarize(self.weight.org)
        # input.data = Quantize(input.data)
        self.weight.data = Ternarize(self.weight.org, self.delta)

        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            self.bias.data = Ternarize(self.bias.org, self.delta) # ternarizza bias
            out += self.bias.view(1, -1).expand_as(out)
        return out


class BinarizeLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input): 
        if not hasattr(self.weight, 'org'): 
            self.weight.org = self.weight.data.clone()  

        # if input.size(1) != 320:  
        input.data = Binarize(input.data)
        # self.weight.data = Binarize(self.weight.org)
        # input.data = Quantize(input.data)
        self.weight.data = Binarize(self.weight.org)  

        out = nn.functional.linear(input, self.weight)  
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()  
            self.bias.data = Binarize(self.bias.org) 
            out += self.bias.view(1, -1).expand_as(out)  
        return out


class TernarizeConv2d(nn.Conv2d):  # TernarizeConv2d
    def __init__(self, *kargs, delta=0.1, **kwargs):
        super(TernarizeConv2d, self).__init__(*kargs, **kwargs)
        self.delta = delta

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        if input.size(1) != 3:  # if on, it doesn't binarize the first layer
            input.data = Ternarize(input.data, self.delta) # ternary activations
            # self.weight.data = Binarize(self.weight.org)
            # input.data = Quantize(input.data)
        self.weight.data = Ternarize(self.weight.org, self.delta)

        out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            self.bias.data = Ternarize(self.bias.org, self.delta) # ternarizza bias
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out


class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input): 
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone() 

        if input.size(1) != 3:  
            input.data = Binarize(input.data)
            # self.weight.data = Binarize(self.weight.org)
            # input.data = Quantize(input.data)
        self.weight.data = Binarize(self.weight.org) 

        out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation,
                                   self.groups) 
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            self.bias.data = Binarize(self.bias.org) 
            out += self.bias.view(1, -1, 1, 1).expand_as(out)  
        return out
