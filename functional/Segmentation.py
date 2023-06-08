from skimage.filters import sobel
from skimage.color import rgb2gray
import cv2
from skimage.segmentation import felzenszwalb, slic, watershed


def watershed_sobel(img, markers, compactness):
    return watershed(sobel(rgb2gray(img)), markers=markers, compactness=compactness)


def SEEDS(image, num_superpixels=50, num_levels=15, n_iter=10):
    seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2],
                                               num_superpixels=num_superpixels,
                                               num_levels=num_levels,
                                               histogram_bins=5,
                                               double_step=True)
    seeds.iterate(image, n_iter)
    label_seeds = seeds.getLabels()
    return label_seeds


def LSC(image, region_size=20, ratio=0.075, n_iter=10):
    lsc = cv2.ximgproc.createSuperpixelLSC(image, region_size=region_size, ratio=ratio)
    lsc.iterate(n_iter)
    label_seeds = lsc.getLabels()
    return label_seeds