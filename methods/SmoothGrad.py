import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import cv2
import torch.nn.functional as F
from torch.autograd import grad
from functional.imagenet_helper import ILSVRC_UTILS


def normalize(X, mu, std):
    return (X - mu)/std


class SmoothGrad(nn.Module):
    def __init__(self, net, shape=(1, 3, 224, 224), sample_size=50,
                 std_level=0.1, device=None):
        super().__init__()

        self.net = net
        self.x_shape = shape
        # variables
        self.samples = sample_size
        self.std_level = std_level
        self.device = device

    def forward(self, x, target_cls=None, sec_ord=False):
        if target_cls == None:
            target_cls = self.net(x).max(1)[1].item()
        elif isinstance(target_cls, torch.Tensor):
            target_cls = target_cls.item()
        self._reset(x, sec_ord)
        batch_x = self._generate_noised_x(x)
        accus = self._predict(batch_x)
        sal = grad(accus[:, target_cls].mean(), x, create_graph=sec_ord)[0]
        accu = self._predict(x)
        return sal, accu

    def _reset(self, x, sec_ord):
        self.x_std = (x.max() - x.min()) * self.std_level
        if not sec_ord:
            x.requires_grad_(True)

    def _predict(self, x):
        return F.softmax(self.net(x), 1)

    def _generate_noised_x(self, x):
        noise = torch.empty(
            self.samples, self.x_shape[1], self.x_shape[2], self.x_shape[3]).normal_(0, self.x_std.item())
        return x + noise.to(self.device)


def smoothGrad(model: nn.Module, inp, target=None, config={}, device='cuda'):
    sm = SmoothGrad(model, **config, device=device)
    out, _ = sm(inp, target)
    return out


def double_percentile(grad):
    grad_max = np.percentile(grad, 99, axis=(2, 3))[:, :, np.newaxis, np.newaxis]
    grad_min = np.percentile(grad, 1, axis=(2, 3))[:, :, np.newaxis, np.newaxis]
    grad[grad < grad_min] = 0.
    grad[grad > grad_max] = 0.
    return grad

@torch.enable_grad()
def smooth_grad(model: nn.Module, inp: torch.Tensor, target=None, loss_fn=None,
                with_normalize=False, mean=None, std=None,
                n_samples=50, stdev=0.15, mode='c', device='cuda:0'):
    if with_normalize:
        assert (mean is not None) and (std is not None)
    x = inp.data.cpu().numpy()
    stdev = stdev * (np.max(x) - np.min(x))
    total_gradients = np.zeros_like(x)
    for i in range(n_samples):
        noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
        x_plus_noise = x + noise
        x_plus_noise = np.clip(x_plus_noise, 0., 1.)
        x_plus_noise = Variable(torch.from_numpy(x_plus_noise).to(inp.device), requires_grad=True)
        if not with_normalize:
            output = model(x_plus_noise)
        else:
            output = model(normalize(x_plus_noise, mean, std))
        if target is None:
            target = torch.argmax(output, dim=1)
        if loss_fn is None:
            one_hot = torch.zeros(output.size()).to(device)
            one_hot[range(output.size(0)), target] = 1
            loss = torch.sum(one_hot * output)
        else:
            loss = loss_fn(output, target)
        loss.backward()
        grad = x_plus_noise.grad.data.cpu().numpy()

        if mode == 'm':
            total_gradients += grad * grad
        else:
            total_gradients += grad
    return total_gradients / n_samples


@torch.no_grad()
def optim_use_smoothgrad(model, inp, target, alpha=.2, n_steps=100, with_tqdm=True):
    processed = inp.detach().clone()
    iteration = tqdm(range(n_steps)) if with_tqdm else range(n_steps)
    for _ in iteration:
        grad = smooth_grad(model, processed, target, with_normalize=True,
                           mean=ILSVRC_UTILS.mean_tensor.cuda(), std=ILSVRC_UTILS.std_tensor.cuda())
        # grad = double_percentile(grad)
        grad = cv2.blur(grad[0].transpose((1, 2, 0)), (3, 3)).transpose((2, 0, 1))[np.newaxis, :]
        processed.data -= -torch.from_numpy(alpha * grad).cuda()
        processed.data = torch.clamp(processed.data, 0., 1.)
        if with_tqdm:
            iteration.set_description('{}'.format(
                model(normalize(processed, ILSVRC_UTILS.mean_tensor.cuda(), ILSVRC_UTILS.std_tensor.cuda()))
                [range(processed.size(0)), target].sum()))
    return processed


if __name__ == '__main__':
    import os
    # import torchvision.models as models
    # from PIL import Image
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from ExplainCV.DataSet.imagenet2012_helper import ILSVRC_UTILS
    # from PIL import Image
    # from ExplainCV.interpreter.configs.option import options
    #
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    #
    # model = models.vgg19(pretrained=True)
    # model = model.eval().cuda()
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # img = Image.open(r'D:\Project\interpreter_mia\images\n01944390_11722.JPEG')
    # img = img.resize((224, 224))
    #
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    #
    # x = np.array(img)
    # x = torch.tensor(x)[None, :].permute((0, 3, 1, 2)) / 225
    # y = torch.LongTensor([988])
    #
    # opts = options(r'D:\Project\interpreter_mia\ExplainCV\interpreter\configs\default_configs.xml')
    # image = smoothGrad(model, x.detach().cuda(), y.cuda(), opts.get('SmoothGrad'))
    # image = image[0].detach().cpu().permute((1, 2, 0)).numpy()
    # image = (image - image.min()) / (image.max() - image.min())
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

    # loss_fn = nn.CrossEntropyLoss()
    #
    # img = Image.open(r'D:\Project\interpreter_mia\images\n02981792_19268.JPEG')
    # img = img.resize((224, 224))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    #
    # x = np.array(img)
    # x = torch.tensor(x)[None, :].permute((0, 3, 1, 2)) / 225
    # y = torch.LongTensor([14])
    # x, y = x.cuda(), y.cuda()
    #
    # from ExplainCV.interpreter.grad_utils import get_gradient
    # from ExplainCV.loss_utils.onehot_loss import onehot_loss
    #
    # # grad = get_gradient(model, x, y, onehot_loss())
    # # grad = grad.detach().cpu().numpy()
    #
    #
    # grad = smooth_grad(model, x, y,
    #                    with_normalize=False, mean=ILSVRC_UTILS.mean_tensor.cuda(), std=ILSVRC_UTILS.std_tensor.cuda(),
    #                    mode='m', device=x.device)
    #
    # grad = grad.transpose(0, 2, 3, 1)
    # grad = (grad - grad.min()) / (grad.max() - grad.min())
    # plt.imshow(np.mean(grad[0], axis=2), cmap='gray')
    # plt.axis('off')
    # plt.show()