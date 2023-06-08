import torch.nn as nn
import torchvision.transforms as tf
from PIL import Image
import numpy as np


class Identity:
    def __init__(self):
        self.f = nn.Identity()
        self.b = nn.Identity()

    def forward(self, x):
        return self.f(x)

    def backward(self, x):
        return self.b(x)


class Rotation:
    def __init__(self, degree):
        self.toTensor = tf.ToTensor()
        self.forward_rotate = getattr(Image, 'ROTATE_'+str(degree))
        self.backward_rotate = 360 - degree

    def forward(self, x):
        img = Image.fromarray(np.uint8(255 * x[0].permute((1, 2, 0)).detach().cpu().numpy()))
        img = img.transpose(self.forward_rotate)
        return self.toTensor(img).unsqueeze(0)

    def backward(self, sal):
        if self.backward_rotate == 270:
            sal = np.transpose(sal[::-1, ...][:, ::-1], axes=(1, 0))[::-1, ...]
        elif self.backward_rotate == 180:
            sal = sal[::-1, ...][:, ::-1]
        elif self.backward_rotate == 90:
            sal = np.transpose(sal, axes=(1, 0))[::-1, ...]
        return sal


class Shift:
    def __init__(self, n_pixel=(0, 0)):
        self.toTensor = tf.ToTensor()
        self.n_pixel = n_pixel

    def forward(self, x):
        img = np.uint8(255 * x[0].detach().cpu().permute((1, 2, 0)).numpy())
        img = np.concatenate([img[self.n_pixel[0]:, :], img[:self.n_pixel[0], :]], axis=0)
        img = np.concatenate([img[:, self.n_pixel[1]:], img[:, :self.n_pixel[1]]], axis=1)
        return self.toTensor(img).unsqueeze(0)

    def backward(self, sal):
        reverse_pixel = (-self.n_pixel[0], -self.n_pixel[1])
        sal = np.concatenate([sal[reverse_pixel[0]:, :], sal[:reverse_pixel[0], :]], axis=0)
        sal = np.concatenate([sal[:, reverse_pixel[1]:], sal[:, :reverse_pixel[1]]], axis=1)
        return sal


class ColorJitter:
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5):
        self.f = tf.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.b = nn.Identity()

    def forward(self, x):
        return self.f(x)

    def backward(self, x):
        return self.b(x)