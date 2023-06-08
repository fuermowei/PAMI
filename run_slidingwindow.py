import torch
import numpy as np


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.models as models
    from methods.sliding_window import SlidingWindow
    import torchvision.transforms as tf
    from timm.data import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN
    from functional.model_utils import freeze_model
    from functional.Blur import blur

    image_path = r'images/ILSVRC2012_val_00038328.JPEG'
    img = Image.open(image_path).convert('RGB').resize((224, 224))

    model = models.vgg19_bn(models.VGG19_BN_Weights.IMAGENET1K_V1).cuda()
    model = freeze_model(model)

    transformer = tf.Compose([
        tf.Resize((224, 224)),
        tf.ToTensor()
    ])
    normalize = tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    image = transformer(img).unsqueeze(0).cuda()
    target = torch.argmax(model(image), dim=1).cuda()
    sw = SlidingWindow(
        window_size=40,
        stride_size=6,
        transformer=normalize,
        batch_size=90,
        substrate=blur(49, 100)  # or torch.zeros_like or torch.ones_like
    )
    importance_map = sw(model, image, target)

    importance_map = importance_map / (sw.N + 1e-9)
    importance_map[-1, -1] = 1
    # importance_map = Image.fromarray(np.uint8(importance_map * 255))

    # importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-9)

    plt.figure()
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[1].imshow(importance_map)
    axes[1].axis('off')
    plt.show()

