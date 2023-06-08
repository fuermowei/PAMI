import torch
from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from functional.imagenet_helper import ILSVRC_UTILS
import torch.nn.functional as F

HW = 224 * 224
n_classes = 1000


def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


# Given label number returns class name
def get_class_name(c):
    return ILSVRC_UTILS.classes[str(c)][1]


def _freeze_model(model: nn.Module):
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen // 2, klen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric:
    def __init__(self, model, mode, step, substrate_fn):
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    @torch.no_grad()
    def single_run(self, img_tensor, explanation, verbose=0, save_to=None):
        r"""Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.
        Return:
            scores (nd.array): Array containing scores at every step.
        """
        pred = self.model(img_tensor.cuda())
        pred = F.softmax(pred, dim=1)
        top, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]
        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1)
        for i in range(n_steps + 1):
            pred = self.model(start.cuda())
            pred = F.softmax(pred, dim=1)
            pr, cl = torch.topk(pred, 2)
            if verbose == 2:
                print('{}: {:.3f}'.format(get_class_name(cl[0][0]), float(pr[0][0])))
                print('{}: {:.3f}'.format(get_class_name(cl[0][1]), float(pr[0][1])))
            scores[i] = pred[0, c]
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')
                tensor_imshow(start[0])

                plt.subplot(122)
                plt.plot(np.arange(i + 1) / n_steps, scores[:i + 1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i + 1) / n_steps, 0, scores[:i + 1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start.cpu().numpy().reshape(1, 3, HW)[0, :, coords] = finish.cpu().numpy().reshape(1, 3, HW)[0, :,
                                                                      coords]
        return scores

    @torch.no_grad()
    def evaluate(self, img_batch, exp_batch, batch_size):
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, n_classes)
        assert n_samples % batch_size == 0
        for i in range(n_samples // batch_size):
            preds = self.model(img_batch[i * batch_size:(i + 1) * batch_size].cuda()).cpu()
            preds = F.softmax(preds, dim=1)
            predictions[i * batch_size:(i + 1) * batch_size] = preds
        top = np.argmax(predictions, -1)
        n_steps = (HW + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(exp_batch.reshape(-1, HW), axis=1), axis=-1)
        r = np.arange(n_samples).reshape(n_samples, 1)
        substrate = torch.zeros_like(img_batch)
        for j in range(n_samples // batch_size):
            substrate[j * batch_size:(j + 1) * batch_size] = self.substrate_fn(
                img_batch[j * batch_size:(j + 1) * batch_size])
        if self.mode == 'del':
            start = img_batch.clone()
            finish = substrate
        elif self.mode == 'ins':
            start = substrate
            finish = img_batch.clone()
        for i in range(n_steps + 1):
            for j in range(n_samples // batch_size):
                preds = self.model(start[j * batch_size:(j + 1) * batch_size].cuda())
                preds = F.softmax(preds, dim=1)
                preds = preds.cpu().numpy()[range(batch_size), top[j * batch_size:(j + 1) * batch_size]]
                scores[i, j * batch_size:(j + 1) * batch_size] = preds
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            start.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords] = \
                finish.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords]
        # print('AUC: {}'.format(auc(scores.mean(1))))
        return scores


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    import torchvision.models as models
    import torchvision.transforms as tf
    from glob import glob
    from PIL import Image
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    save_step = 10

    model = _freeze_model(models.vgg19_bn(models.VGG19_BN_Weights.IMAGENET1K_V1)).cuda()

    klen = 49
    ksig = 100
    kern = gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen // 2)

    insertion = CausalMetric(model, 'ins', 224, substrate_fn=blur)
    deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)

    data_path = r'/data/Public/Datasets/ilsvrc2012/val/'
    heatmap_root = r"/data/shiwei/ICLR2023/AUG/AUG_VGG19BN_TWOSTAGE/"

    save_path = r'saver'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    sub_class = os.listdir(heatmap_root)
    transformer = tf.Compose([
        tf.Resize((224, 224)),
        tf.ToTensor(),
        tf.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    scores = np.zeros((225, 1))
    past = 0
    verbose = tqdm(sub_class)

    step = 0
    for cls in verbose:
        path = os.path.join(heatmap_root, cls)
        target = torch.LongTensor([ILSVRC_UTILS.labels_int[cls]]).cuda()
        image_list, file_list = [], []
        for file in glob(os.path.join(path, '*.npy')):
            img = Image.open(os.path.join(data_path, cls, os.path.basename(file).split('.')[0] + '.JPEG')).convert(
                'RGB')
            image_list.append(transformer(img).unsqueeze(0).cuda())
            file_list.append(np.load(file)[None, :, :])

        data = np.concatenate(file_list, axis=0)
        batch = torch.concat(image_list, dim=0)
        predict = model(batch)

        flag = (torch.argmax(predict, dim=1) == target).cpu().numpy()
        batch = batch[flag]
        data = data[flag]
        target = target.repeat(batch.size(0))
        size = batch.size(0)

        ret = insertion.evaluate(batch.cpu(), data, batch_size=size)
        scores = (scores * past + np.sum(ret, axis=1, keepdims=True)) / (past + size)
        past += size
        verbose.set_description('{}'.format(auc(scores.mean(1))))

        if step % save_step == 0:
            np.save(os.path.join(save_path, 'aug_insertion2.npy'), {'step': step, 'past': past, 'scores': scores})
        step += 1

    # np.save(os.path.join(save_path, 'aug_insertionn2.npy'), {'step': step, 'past': past, 'scores': scores})
