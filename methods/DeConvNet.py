import torch
from torch.nn import ReLU
from torch.autograd import Variable

class Deconv():
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        if self.model.__class__.__name__ == 'ResNet':
            first_layer = self.model.conv1
        else:
            first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        def relu_backward_hook_function(module, grad_in, grad_out):
            modified_grad_out = torch.clamp(grad_in[0], min=0.0)
            return (modified_grad_out,)
        for module in list(self.model.modules()):
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)

    @torch.enable_grad()
    def generate_gradients(self, input_image, target_class):
        input_image = Variable(input_image, requires_grad=True)
        model_output = self.model(input_image)
        self.model.zero_grad()
        one_hot_output = torch.FloatTensor(input_image.size(0),
                                           model_output.size()[-1]).zero_().to(input_image.device)
        one_hot_output[range(input_image.size(0)), target_class] = 1
        model_output.backward(gradient=one_hot_output)
        gradients_as_arr = self.gradients.data.cpu().numpy()
        return gradients_as_arr


if __name__ == '__main__':
    import os
    import torchvision.models as models
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    model = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V2).eval().cuda()
    # model = models.vgg16(models.VGG16_Weights.IMAGENET1K_V1).eval().cuda()
    for param in model.parameters():
        param.requires_grad = False
    gbp = Deconv(model)

    img = Image.open(r'D:\Project\interpreter_mia\images\n02981792_19047.JPEG')
    img = img.resize((224, 224))
    # t = copy.deepcopy(img)

    plt.imshow(img)
    plt.axis('off')
    plt.show()

    x = np.array(img)
    x = torch.tensor(x)[None, :].permute((0, 3, 1, 2)) / 225
    y = torch.argmax(model(x.cuda()), dim=1)

    image = gbp.generate_gradients(x.cuda(), y.cuda())

    #
    # img = Image.open(r'D:\Project\interpreter_mia\images\n01749939_1007.JPEG')
    # img = img.resize((224, 224))
    # x = np.array(img)
    # x = torch.tensor(x)[None, :].permute((0, 3, 1, 2)) / 225
    # y = torch.LongTensor([988])
    #
    # image = gbp.generate_gradients(x.cuda(), y.cuda())
    print(image)
    image = np.max(image, axis=0)
    image = (image - image.min()) / (image.max() - image.min() + 1e-9)
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.show()

    # plt.figure()
    # _, heatmap = apply_colormap_on_image(t, image, alpha=0.6)
    # plt.imshow(heatmap)
    # plt.show()