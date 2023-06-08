import copy
import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from functional.model_utils import freeze_model


class PreSegmentation:
    def __init__(self, transformer=nn.Identity(), mini_batch=32,
                 substrate=torch.zeros_like, augmentation=None, segmentation=None):
        self.image, self.target, self.model, self.img = None, None, None, None
        if augmentation is None:
            self.augmentation = [nn.Identity()]
        else:
            self.augmentation = augmentation
        self.segmentation = segmentation
        self.transformer = transformer
        self.mini_batch = mini_batch
        self.substrate = substrate
        self.N = None

    @torch.no_grad()
    def get_score_dilate(self, slice_batch):
        slice = np.uint8(slice_batch)
        slice_bool = ~(slice.astype(np.bool))
        slice = torch.from_numpy(slice).type(torch.FloatTensor).to(self.image.device)
        bkg = torch.from_numpy(np.uint8(slice_bool)).type(torch.FloatTensor).to(self.image.device)
        image = self.image.repeat(np.shape(slice)[0], 1, 1, 1) * slice[:, None, :, :] + \
                self.bkg.repeat(np.shape(slice)[0], 1, 1, 1) * bkg[:, None, :, :]
        output = self.model(self.transformer(image))
        output = F.softmax(output, dim=1)
        return output

    def segmentation_heatmap(self):
        masks = []
        for idx, (method, config) in enumerate(self.segmentation):
            split = method(self.candidate, **config)
            if split.min() == 0:
                split += 1
            for i in range(1, split.max() + 1):
                mask = split == i
                masks.append(mask[None, :, :])
        masks = np.concatenate(masks, axis=0)
        self.N += np.sum(masks, axis=0)
        heatmap = np.zeros((self.image.size(2), self.image.size(3)))
        for i in range(0, np.shape(masks)[0], self.mini_batch):
            batch = masks[i:min(i + self.mini_batch, np.shape(masks)[0])]
            if len(np.shape(batch)) == 2:
                batch = batch[None, :, :]
            scores = self.get_score_dilate(batch)[:, self.target[0]]
            heatmap += np.sum(scores.detach().cpu().numpy()[:, None, None] * batch, axis=0)
        return heatmap

    def __call__(self, model, image, target, heatmap=None):
        self.N = np.zeros((image.size(2), image.size(3)))
        if heatmap is None:
            self.model, self.ori, self.target = freeze_model(model), image, target
            heatmap = np.zeros((self.ori.size(2), self.ori.size(3)))
            for aug in self.augmentation:
                self.image = aug.forward(self.ori).to(image.device)
                self.candidate = np.uint8(255 * self.image[0].cpu().permute((1, 2, 0)).numpy())
                self.bkg = self.substrate(self.image)
                ht = self.segmentation_heatmap()
                ht = aug.backward(ht)
                heatmap += ht
        else:
            self.model, self.ori, self.target = freeze_model(model), image, target
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9)
            color_map = plt.get_cmap('viridis')
            self.heatmap_one = color_map((heatmap * 255).astype(np.uint8))[:, :, :3]
            heatmap = np.zeros((self.ori.size(2), self.ori.size(3)))
            for aug in self.augmentation:
                self.image = aug.forward(self.ori).to(image.device)
                self.candidate = aug.forward(self.toTensor(self.heatmap_one).unsqueeze(0)).to(image.device)
                self.candidate = np.uint8(255 * self.candidate[0].cpu().permute((1, 2, 0)).numpy())
                self.bkg = self.substrate(self.image)
                ht = self.segmentation_heatmap()
                ht = aug.backward(ht)
                heatmap += ht
        return heatmap


