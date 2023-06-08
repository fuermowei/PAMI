import math
import random
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from numbers import Number
import torch


class DummyPbar():
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        return

    def close(self, *args, **kwargs):
        return


class ExplanationMethod(torch.nn.Module):
    def __init__(self, model, preprocess, postprocess):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess

    def process(self, inputs, target):
        raise NotImplementedError

    def _forward(self, inputs, target=None):
        if target is None:
            target = self.model(inputs).argmax(1)
        if isinstance(target, Number):
            target = [target]
        if not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target)
        out = inputs
        if self.preprocess is not None:
            out = self.preprocess(inputs)

        if out.device != target.device:
            target = target.to(out.device)
        out = self.process(out, target)
        if self.postprocess is not None:
            out = self.postprocess(out)
        return out

    forward = _forward


class RandomMaskSampler(Dataset):
    def __init__(self, nmasks, cell_size, image_size, p=0.5, half=True):
        self.nmasks = nmasks
        self.image_size = image_size
        self.cellsize = cell_size
        self.p = p
        self.half = half

    def initial_explanation(self, nclasses):
        x = torch.zeros(nclasses, self.image_size, self.image_size)
        if self.half:
            x = x.half()
        return x

    def __len__(self):
        return self.nmasks

    def __getitem__(self, idx):
        grid_size = (math.ceil(self.image_size[0] / self.cellsize), math.ceil(self.image_size[1] / self.cellsize))
        upsize = ((self.cellsize+1) * grid_size[0], (self.cellsize+1) * grid_size[1])
        mask = torch.rand(1, 1, self.cellsize, self.cellsize,
                          dtype=torch.half if self.half else None)
        if self.half:
            mask = (mask < self.p).half()
        else:
            mask = (mask < self.p).float()
        mask = F.pad(mask, (1, 1, 1, 1), mode='reflect')
        mask = F.interpolate(mask, size=(upsize[0]+grid_size[0]*2, upsize[1]+grid_size[1]*2),
                             mode='bilinear', align_corners=False)
        mask = mask[:, :, grid_size[0]:-grid_size[0], grid_size[1]:-grid_size[1]]

        yshift = random.randrange(0, grid_size[0])
        xshift = random.randrange(0, grid_size[1])
        mask = mask.squeeze(0)
        mask = mask[:,
                    yshift:yshift+self.image_size[0],
                    xshift:xshift+self.image_size[1]]
        return mask


class RISE(ExplanationMethod):
    def __init__(self, model, num_masks=8000, cell_size=7,
                 probability=0.5, batch_size=1000, progress=False,
                 preprocess=None, postprocess=None):
        super().__init__(model, preprocess, postprocess)
        self.nmasks = num_masks
        self.cell_size = cell_size
        self.p = probability
        self.batch_size = batch_size
        self.get_pbar = tqdm if progress else DummyPbar

    @torch.no_grad()
    def process(self, inputs, targets):
        device = inputs.device
        imsize = inputs.shape[2:]
        half = (inputs.dtype == torch.half)
        masksampler = RandomMaskSampler(self.nmasks, self.cell_size, imsize,
                                        self.p, half=half)
        maskloader = DataLoader(masksampler, batch_size=self.batch_size)
        explanations = torch.zeros(inputs.size(0), 1, imsize[0], imsize[1],
                                   dtype=torch.half if half else None,
                                   device=device)

        pbar = self.get_pbar(total=inputs.size(0)*len(maskloader), ncols=80)
        for input, explanation, target in zip(inputs, explanations, targets):
            for random_mask in maskloader:
                random_mask = random_mask.to(device)
                prob = self.model(input * random_mask).softmax(1)
                importance = torch.tensordot(
                    prob, random_mask, [[0], [0]]).squeeze()
                explanation.add_(importance[target])
                pbar.update()
        pbar.close()

        return explanations.div_(self.nmasks).div_(self.p)