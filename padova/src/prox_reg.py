import torch
import torch.nn.functional as F


def if_binary(n):
    return not ('bn' in n) and not ('conv1' in n)
    # return not ('bn' in n)


class BinOp():
    def __init__(self, model, if_binary=if_binary):
        self.model = model
        self.saved_params = {}
        self.init_params = {}
        self.if_binary = if_binary
        for n, p in model.named_parameters():
            if self.if_binary(n):
                self.saved_params[n] = p.data.clone()
                self.init_params[n] = p.data.clone()

    def prox_operator(self, reg, reg_type='binary'):
        if reg_type == 'binary':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    '''
                    # ProxQuant
                    p_sign, p_abs = p.data.sign(), p.data.abs()
                    p.data.copy_(p_sign * (F.relu((p_abs - 1).abs() - reg) * (p_abs - 1).sign() + 1))
                    '''
                    # ConQ (ours)
                    c1 = torch.abs(p) < (1 - 2 * reg)
                    v1 = p / (1 - 2 * reg)
                    c2 = torch.abs(p) > (1 + reg)
                    v2 = p - reg * torch.sign(p)
                    c3 = torch.logical_and(torch.abs(p) >= (1 - 2 * reg), torch.abs(p) <= (1 + reg))
                    v3 = torch.sign(p)
                    t1 = torch.where(c1, v1, p)
                    t2 = torch.where(c2, v2, t1)
                    t3 = torch.where(c3, v3, t2)
                    p.data.copy_(t3)

    def quantize(self, mode='deterministic'):
        if mode == 'deterministic':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p.data.copy_(p.data.sign())
        elif mode == 'binary_freeze':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p.data.copy_(p.data.sign())
                    p.requires_grad = False

    def quantize_error(self, mode='deterministic', n_bits=1, by_row=False, n_rounds=1):
        self.save_params()
        self.quantize(mode=mode)
        results = {'quant_error_' + n: (p.data - self.saved_params[n]).norm() / self.saved_params[n].norm()
                   for n, p in self.model.named_parameters() if self.if_binary(n)}
        self.restore()
        return results

    def clip(self, clip_type='binary'):
        if clip_type == 'binary':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p.data.clamp_(-1, 1)

    def save_params(self):
        for n, p in self.model.named_parameters():
            if self.if_binary(n):
                self.saved_params[n].copy_(p.data)

    def restore(self):
        for n, p in self.model.named_parameters():
            if self.if_binary(n):
                p.data.copy_(self.saved_params[n])
