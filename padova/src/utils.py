from torch import nn


def stash_model_parameters(model: nn.Module):
    for param in model.parameters():
        setattr(param, 'original', param.data.clone())


def unstatsh_model_parameters(model: nn.Module):
    for param in model.parameters():
        if hasattr(param, 'original'):
            param.data.copy_(param.original)


def clamp_model_parameters(model: nn.Module):
    for param in model.parameters():
        if hasattr(param, 'original'):
            data_clamped = param.data.clamp_(-1, 1)
            param.original.copy_(data_clamped)


def hparams_to_folder(hparams: dict) -> str:
    x = [f"{k}={v}" for k, v in hparams.items()]
    return "_".join(x)
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)