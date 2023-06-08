import numpy as np
import torch
import torch.nn as nn
from functional.model_utils import freeze_model
import torch.nn.functional as F


class CircleMask:
    def __init__(self, radius, stride):
        self.r = radius
        self.stride = (stride, stride)

    def __call__(self, size):
        imap = []
        for i in range(0, size[0], self.stride[0]):
            for j in range(1, size[1], self.stride[1]):
                y, x = np.ogrid[:size[0], :size[1]]
                y = y[::-1]
                mask = ((x - i) ** 2 + (y - j) ** 2) <= self.r ** 2
                imap.append(torch.from_numpy(mask).type(torch.bool).unsqueeze(0))
        imap = torch.cat(imap, dim=0)
        return imap.squeeze()


class SlidingWindow:
    def __init__(self, window_size=40, stride_size=6, batch_size=90,
                 transformer=nn.Identity(), substrate=torch.zeros_like):
        self.generator = CircleMask(window_size, stride_size)
        self.transformer = transformer
        self.substrate = substrate
        self.batch_size = batch_size
        self.N = None

    @torch.no_grad()
    def __call__(self, model, image, target):
        self.masks = self.generator((image.size(2), image.size(3))).cuda()
        model = freeze_model(model)
        image, target = image.cuda(), target.cuda()
        self.N = np.zeros((image.size(2), image.size(3)))
        bkg = self.substrate(image).cuda()
        saliency = np.zeros((image.size(2), image.size(3)))
        for b in range(0, self.masks.size(0), self.batch_size):
            mask = self.masks[b:min(b + self.batch_size, self.masks.size(0))].unsqueeze(1)
            self.N += np.sum(mask[:, 0, :, :].detach().cpu().numpy(), axis=0)
            inp = image.repeat(mask.size(0), 1, 1, 1)
            inp = inp * mask + bkg * (~mask)
            output = model(self.transformer(inp))
            output = F.softmax(output, dim=1)[:, target[0]]
            saliency += np.sum((output[:, None, None, None] * mask).detach().cpu().numpy(), axis=0)[0]
        return saliency