from scipy.ndimage.filters import gaussian_filter
import torch
import torch.nn.functional as F


class blur:
    def __init__(self, klen, nsig):
        self.kern = self.gkern(klen, nsig).cuda()
        self.klen = klen

    def gkern(self, klen, nsig):
        import numpy as np
        inp = np.zeros((klen, klen))
        inp[klen // 2, klen // 2] = 1
        k = gaussian_filter(inp, nsig)
        kern = np.zeros((3, 3, klen, klen))
        kern[0, 0] = k
        kern[1, 1] = k
        kern[2, 2] = k
        return torch.from_numpy(kern.astype('float32'))

    def __call__(self, x):
        return F.conv2d(x, self.kern, padding=self.klen // 2)