import torch.nn as nn


def freeze_model(model: nn.Module):
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model