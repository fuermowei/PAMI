# PAMI
This is the repository for Pytorch Implementation of "Pami: Partition input and Aggregate outputs for Model Interpretation". You are welcome to raise any questions or concerns about this repository in this issue, and the authors will respond as soon as possible.

## Modules

## Requirements
- Install pytorch, torchvision
- Install skimage, cv2, scipy

## Method
![MainMap](https://github.com/fuermowei/PAMI/assets/47769416/65d4a763-936a-433d-b19f-9cd5934d85fd)

## Step 1: Model preparation
Loading a pre-trained model and adjusting it to inference mode. Meanwhile, freezing the parameter updates can effectively reduce the computation time.
```
model = models.vgg19_bn(models.VGG19_BN_Weights.IMAGENET1K_V1).eval()
for param in model.parameters():
    param.requires_grad = False
model = model.cuda()
```
## Step 2
