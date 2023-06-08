import numpy as np
import torch
import torch.nn as nn
import os
from captum.attr import Occlusion
import torch.nn.functional as F

torch.manual_seed(123)
np.random.seed(123)



class wrapped_model(nn.Module):
    def __init__(self, model):
        super(wrapped_model, self).__init__()
        self.model = model
    def forward(self, inp):
        return F.softmax(self.model(inp), dim=1)


def _freeze_model(model: nn.Module):
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


class OCC:
    def __init__(self, model, strides=6, sliding_window_shapes=(3, 70, 70)):
        self.model = _freeze_model(wrapped_model(model)).cuda()
        self.occ = Occlusion(self.model)
        self.strides = strides
        self.sliding_window_shapes = sliding_window_shapes

    def process(self, image, target):
        return self.occ.attribute(image,
                           strides=self.strides,
                           sliding_window_shapes=self.sliding_window_shapes,
                           target=target)


if __name__ == '__main__':
    import captum
    import torchvision.models as models
    import torchvision.transforms as tf
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from PIL import Image
    from functional.imagenet_helper import ILSVRC_UTILS
    import matplotlib.pyplot as plt

    path = '/data/shiwei/ilsvrc2012/val/n04118776/ILSVRC2012_val_00022556.JPEG'
    img = Image.open(path).convert('RGB')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    normalization = tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    transformer = tf.Compose([
        tf.Resize((224, 224)),
        tf.ToTensor(),
        normalization
    ])
    image = transformer(img).unsqueeze(0).cuda()
    model = models.vgg19_bn(models.VGG19_BN_Weights.IMAGENET1K_V1).cuda()
    model = _freeze_model(model).cuda()
    occ = OCC(model)

    mp = occ.process(image, torch.LongTensor([ILSVRC_UTILS.labels_int['n02027492']]).cuda())
    mp = torch.mean(mp, dim=1).squeeze(1)
    mp = (mp - mp.min()) / (mp.max() - mp.min() + 1e-9)
    plt.imshow(mp[0].detach().cpu().numpy())
    plt.axis('off')
    plt.show()