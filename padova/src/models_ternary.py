import torch
import torch.nn as nn
import math


class CNN_ternary(nn.Module):
    def __init__(self, numClasses, layer_inflation=1, delta=0.1, f32_activations=False):
        super(CNN_ternary, self).__init__()

        def infl(x: int):
            return round(x*layer_inflation)
        
        ch_l1 = 32
        ch_l2 = 64
        ch_l3 = 128
        ch_fc = 2048

        # dropout = 0.1
    
        self.net = nn.Sequential(

            # First layer: Convolutional layer with kernel=[1,9], stride=[0,4]: form 1*6*128 -> 32*6*64
            TernarizeConv2d(1, ch_l1, kernel_size=(1, 9), stride=(1, 2), padding=(0, 4), delta=delta, is_first=True, f32_activations=f32_activations),
            nn.BatchNorm2d(ch_l1),
            # nn.Dropout2d(p=dropout),
            nn.Hardtanh(inplace=True),

            # Second layer: Pool layer with kernel=[1,2], stride=[1,2]: form 32*6*64 -> 32*6*32
            nn.MaxPool2d(kernel_size=[1, 2], stride=(1, 2)),

            # Third layer: Convolutional layer with kernel=[1,3], stride=1: form 32*6*32 -> 64*6*32
            TernarizeConv2d(ch_l1, ch_l2, kernel_size=(1, 3), stride=1, padding=(0, 1), delta=delta, f32_activations=f32_activations),
            nn.BatchNorm2d(ch_l2),
            # nn.Dropout2d(p=dropout),
            nn.Hardtanh(inplace=True),

            # Fourth Layer
            TernarizeConv2d(ch_l2, ch_l3, kernel_size=(1, 3), stride=1, padding=(0, 1), delta=delta, f32_activations=f32_activations),
            nn.BatchNorm2d(ch_l3),
            # nn.Dropout2d(p=dropout),
            nn.Hardtanh(inplace=True),

            # Fifth layer: Pool layer with kernel=[1,2], stride=[1,2]: form 128*6*32 -> 128*6*16
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            TernarizeConv2d(ch_l3, ch_l3, kernel_size=(6, 1), stride=1, padding=0, delta=delta, f32_activations=f32_activations),
            nn.BatchNorm2d(ch_l3),
            # nn.Dropout2d(p=dropout),
            nn.Hardtanh(inplace=True)
        )

        self.fc = TernarizeLinear(ch_fc, numClasses, delta=delta, f32_activations=f32_activations)

        # init_weights(self)

    def forward(self, x):
        # forward operation: propagates the input though the the layers predicting the result
        out = self.net(x)
        out = out.reshape(out.size(0), -1)  # flat the matrix into an array
        out = self.fc(out)
        return out


    def set_delta(self, delta: float):
        for m in self.modules(): 
            if isinstance(m, TernarizeConv2d):  
                m.delta = delta
            elif isinstance(m, TernarizeLinear):
                m.delta = delta

    
    def stats(self):
        zeros = 0
        plus_ones = 0
        minus_ones = 0

        for p in self.parameters():
            if p is not None:
                zeros += torch.sum(torch.eq(p, 0)).item()
                plus_ones += torch.sum(torch.eq(p, 1)).item()
                minus_ones += torch.sum(torch.eq(p, -1)).item()

        num_params = sum([p.nelement() for p in self.parameters()])
        return (zeros, plus_ones, minus_ones, num_params)
    

    def get_weights_entropy(self):
        zeros, plus_ones, minus_ones, num_params = self.stats()
        ps = torch.tensor([zeros, plus_ones, minus_ones]) / num_params
        return -1 * (ps * torch.log2(ps + 1e-9)).sum().item()



def init_weights(model):  
    for m in model.modules(): 

        if isinstance(m, TernarizeConv2d):  
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  
            m.weight.data.normal_(0, math.sqrt(2. / n))

        elif isinstance(m, TernarizeLinear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()



class TernarizeLinear(nn.Linear):  # TernarizeLinear
    def __init__(self, *kargs, delta=0.1, f32_activations=False, **kwargs):
        super(TernarizeLinear, self).__init__(*kargs, **kwargs)
        self.delta = delta
        self.f32_activations = f32_activations

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        max_w = torch.max(self.weight.org)
        d = (self.delta * max_w).item()

        if not self.f32_activations:
            input.data = Ternarize(input.data, d) # ternary activations
            # print('binarizing [linear]')
        self.weight.data = Ternarize(self.weight.org, d)

        out = nn.functional.linear(input, self.weight)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            self.bias.data = Ternarize(self.bias.org, d) # ternarizza bias
            out += self.bias.view(1, -1).expand_as(out)

        return out


class TernarizeConv2d(nn.Conv2d):  # TernarizeConv2d
    def __init__(self, *kargs, delta=0.1, is_first=False, f32_activations=False, **kwargs):
        super(TernarizeConv2d, self).__init__(*kargs, **kwargs)
        self.delta = delta
        self.is_first = is_first
        self.f32_activations = f32_activations

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        max_w = torch.max(self.weight.org)
        d = (self.delta * max_w).item()

        if not (self.is_first or self.f32_activations):  # doesn't ternarizza the first layer
            input.data = Ternarize(input.data, d) # ternary activations
            # print('binarizing [conv2d]')
        self.weight.data = Ternarize(self.weight.org, d)

        out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            self.bias.data = Ternarize(self.bias.org, d) # ternarizza bias
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


def Ternarize(tensor, delta):
    cond1 = torch.abs(tensor) < delta
    cond2 = tensor >= delta
    cond3 = tensor <= -delta
    t1 = torch.where(cond1, torch.tensor(0.).cuda(), tensor)
    t2 = torch.where(cond2, torch.tensor(1.).cuda(), t1)
    t3 = torch.where(cond3, torch.tensor(-1.).cuda(), t2)
    return t3
