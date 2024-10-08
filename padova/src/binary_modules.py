import torch
import torch.nn as nn


def Binarize(tensor):
    return tensor.sign()


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, af32=False, **kwargs):
        self.af32 = af32
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 1 and not self.af32:
            # print("bin")
            input.data = Binarize(input.data)

        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        if input.size(1) != 1:
            self.weight.data = Binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            self.bias.data = Binarize(self.bias.data)
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, af32=False, **kwargs):
        self.af32 = af32
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        # if input.size(1) != 1:
        if not self.af32:
            # print("bin")
            input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            self.bias.data = Binarize(self.bias.data)
            out += self.bias.view(1, -1).expand_as(out)
        return out



class BinaryHammingLoss(nn.Module):
    def __init__(self, *kargs, **kwargs):
        super(BinaryHammingLoss, self).__init__(*kargs, **kwargs)

    def forward(self, x, y):
        x.data = Binarize(x.data)
        y.data = Binarize(y.data)

        return torch.abs(x - y).sum(dim=1) / x.size(1) / 2