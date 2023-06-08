import copy

from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
# from ExplainCV.image_utils.heatmap import apply_colormap_on_image


def distangle_by_name(model, submodule_key):
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


class CamExtractor():
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.module_out = None
        self.register_module()

    def save_gradient(self, grad):
        self.gradients = grad

    def register_module(self):
        def hook(module, inp, out):
            self.module_out = out
            self.module_out.register_hook(self.save_gradient)

        module = distangle_by_name(self.model, self.target_layer)
        module.register_forward_hook(hook)

    def forward_pass(self, x):
        x = Variable(x, requires_grad=True)
        x = self.model(x)
        return self.module_out, x


class GradCam():
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.extractor = CamExtractor(self.model, target_layer)

    @torch.enable_grad()
    def generate_cam(self, input_image, target_class=None):
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        one_hot_output = torch.FloatTensor(model_output.size(0), model_output.size()[-1]).zero_()
        one_hot_output[range(model_output.size(0)), target_class] = 1
        self.model.zero_grad()
        model_output.backward(gradient=one_hot_output.to(input_image.device), retain_graph=True)
        guided_gradients = self.extractor.gradients.data.cpu().numpy()
        target = conv_output.data.cpu().numpy()
        weights = np.mean(guided_gradients, axis=(2, 3))
        weights = np.maximum(weights, 0)
        cam = np.zeros((target.shape[0], *target.shape[2:]), dtype=np.float32)
        cam += np.sum(weights[:, :, None, None] * target, axis=1)
        cam = (cam - np.min(cam, axis=(1, 2))[:, None, None]) / \
              (np.max(cam, axis=(1, 2))[:, None, None] - np.min(cam, axis=(1, 2))[:, None, None])
        cam = np.uint8(cam * 255)
        ret = None
        for i in range(np.shape(cam)[0]):
            image = np.uint8(Image.fromarray(cam[i]).resize((input_image.shape[3],
                                                             input_image.shape[2]), Image.ANTIALIAS)) / 255
            ret = np.concatenate([ret, image[None, :, :]], axis=0) if ret is not None else image[None, :, :]
        return ret


def cam_for_module_name(model, inp, module_name=None, target=None):
    cam = GradCam(model, module_name)
    heatmap = cam.generate_cam(inp, target)
    return heatmap



final_layer = {
    'vgg16': 'features.30',
    'vgg19': 'features.36',
    'vgg19_bn': 'features.51',
    'alexnet': 'features.12',
    'resnet18': 'layer4',
    'resnet50': 'layer4',
    'resnet101': 'layer4',
    'resnet152': 'layer4',
    'regnet_x_16gf': 'trunk_output',
    'regnet_y_16gf': 'trunk_output',
    'regnet_y_128gf': 'trunk_output'
}


def gradCAM(model, inp, target=None, config=None, device='cuda'):
    cam = GradCam(model, final_layer[config['model_name']])
    heatmap = cam.generate_cam(inp, target)
    return torch.from_numpy(heatmap).squeeze(0)


if __name__ == '__main__':
    import os
    import torchvision.models as models
    # from PIL import Image
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from ExplainCV.DataSet.imagenet2012_helper import ILSVRC_UTILS
    # from PIL import Image
    # from ExplainCV.interpreter.configs.option import options
    #
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # model_name = 'resnet50'
    # model = getattr(models, model_name)(pretrained=True)
    #
    # print(model)
    #
    # model = model.eval().cuda()
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # img = Image.open(r'D:\Project\interpreter_mia\images\n01537544_10193.JPEG')
    # img = img.resize((224, 224))
    # t = copy.deepcopy(img)
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
    # config = opts.get('GradCAM')
    # config.update({'model_name': model_name})
    # image = gradCAM(model, x.detach().cuda(), y.cuda(), config)
    #
    # image = image.detach().cpu().numpy()
    # image = (image - image.min()) / (image.max() - image.min())
    # print(image.shape)
    #
    # plt.imshow(image, cmap=plt.cm.hot)
    # plt.colorbar()
    # plt.axis('off')
    # plt.show()
    #
    # plt.figure()
    # _, heatmap = apply_colormap_on_image(t, image, alpha=0.6)
    # plt.imshow(heatmap)
    # plt.show()

