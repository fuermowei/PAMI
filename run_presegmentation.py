import torch

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.models as models
    from methods.pre_segmentation import PreSegmentation
    import torchvision.transforms as tf
    from timm.data import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN
    from functional.model_utils import freeze_model
    from functional.Blur import blur
    from functional.Augmentation import Identity
    from functional.Augmentation import Shift
    from functional.Segmentation import felzenszwalb, slic, SEEDS, watershed_sobel

    augumentation = [Identity(),
                     Shift((36, 0)), Shift((0, 36)), Shift((-36, 0)), Shift((0, -36)),
                     Shift((30, 0)), Shift((0, 30)), Shift((-30, 0)), Shift((0, -30)),
                     Shift((24, 0)), Shift((0, 24)), Shift((-24, 0)), Shift((0, -24))]

    segmentation = [
            (felzenszwalb, {'scale': 250, 'sigma': 0.8, 'min_size': 28 * 28}),
            (felzenszwalb, {'scale': 200, 'sigma': 0.8, 'min_size': 28 * 28}),
            (felzenszwalb, {'scale': 150, 'sigma': 0.8, 'min_size': 28 * 28}),
            (felzenszwalb, {'scale': 100, 'sigma': 0.8, 'min_size': 28 * 28}),
            (felzenszwalb, {'scale': 70, 'sigma': 0.8, 'min_size': 28 * 28}),
            (felzenszwalb, {'scale': 50, 'sigma': 0.8, 'min_size': 28 * 28}),
            (slic, {'n_segments': 10, 'compactness': 20}),
            (slic, {'n_segments': 20, 'compactness': 20}),
            (slic, {'n_segments': 30, 'compactness': 20}),
            (slic, {'n_segments': 40, 'compactness': 20}),
            (slic, {'n_segments': 50, 'compactness': 20}),
            (slic, {'n_segments': 60, 'compactness': 20}),
            (slic, {'n_segments': 70, 'compactness': 20}),
            (slic, {'n_segments': 80, 'compactness': 20}),
            (SEEDS, {'num_superpixels': 10, 'num_levels': 5, 'n_iter': 10}),
            (SEEDS, {'num_superpixels': 20, 'num_levels': 5, 'n_iter': 10}),
            (SEEDS, {'num_superpixels': 30, 'num_levels': 5, 'n_iter': 10}),
            (watershed_sobel, {'markers': 10, 'compactness': 0.0001}),
            (watershed_sobel, {'markers': 20, 'compactness': 0.0001}),
            (watershed_sobel, {'markers': 30, 'compactness': 0.0001})
        ]

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

    ps = PreSegmentation(
        transformer=normalize,
        mini_batch=90,
        substrate=blur(49, 100),  # or torch.zeros_like or torch.ones_like
        augmentation=augumentation,
        segmentation=segmentation
    )

    importance_map = ps(model, image, target)

    importance_map = importance_map / (ps.N + 1e-9)
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

