import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=False)


def save(mask, img, blurred):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    cam = 1.0 * heatmap + np.float32(img) / 255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)

    cv2.imwrite("perturbated.png", np.uint8(255 * perturbated))
    cv2.imwrite("heatmap.png", np.uint8(255 * heatmap))
    cv2.imwrite("mask.png", np.uint8(255 * mask))
    cv2.imwrite("cam.png", np.uint8(255 * cam))


def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v


def load_model():
    model = models.vgg19(pretrained=True)
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False
    return model


@torch.enable_grad()
def meaningful_purturbation(model, inp, target):
    tv_beta = 3
    learning_rate = 0.1
    max_iterations = 500
    l1_coeff = 0.01
    tv_coeff = 0.2

    img = np.uint8(inp.detach()[0].cpu().permute((1, 2, 0)).numpy() * 255)
    ori_img = img
    img = np.float32(ori_img) / 255
    blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
    blurred_img2 = np.float32(cv2.medianBlur(ori_img, 11)) / 255
    blurred_img_numpy = (blurred_img1 + blurred_img2) / 2
    mask_init = np.ones((28, 28), dtype=np.float32)

    img = preprocess_image(img)
    blurred_img = preprocess_image(blurred_img2)
    mask = numpy_to_torch(mask_init)

    if use_cuda:
        upsample = torch.nn.UpsamplingBilinear2d(size=(inp.size(2), inp.size(3))).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=(inp.size(2), inp.size(3)))
    optimizer = torch.optim.Adam([mask], lr=learning_rate)

    target = torch.nn.Softmax()(model(img))
    category = np.argmax(target.cpu().data.numpy())

    for i in range(max_iterations):
        upsampled_mask = upsample(mask)
        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = \
            upsampled_mask.expand(1, 3, upsampled_mask.size(2), \
                                  upsampled_mask.size(3))

        # Use the mask to perturbated the input image.
        perturbated_input = img.mul(upsampled_mask) + \
                            blurred_img.mul(1 - upsampled_mask)

        noise = np.zeros((inp.size(2), inp.size(3), 3), dtype=np.float32)
        cv2.randn(noise, 0, 0.2)
        noise = numpy_to_torch(noise)
        perturbated_input = perturbated_input + noise

        outputs = torch.nn.Softmax()(model(perturbated_input))
        loss = l1_coeff * torch.mean(torch.abs(1 - mask)) + \
               tv_coeff * tv_norm(mask, tv_beta) + outputs[0, category]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)

    upsampled_mask = upsample(mask)
    return upsampled_mask



