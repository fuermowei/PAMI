# PAMI: Partition input and Aggregate outputs for Model Interpretation
This is the repository for Pytorch Implementation of "Pami: Partition input and Aggregate outputs for Model Interpretation". You are welcome to raise any questions or concerns about this repository in this issue, and the authors will respond as soon as possible.

## Modules

## Requirements
- Install pytorch, torchvision
- Install skimage, cv2, scipy, PIL
- Install timm (optional)

## Method
![MainMap](https://github.com/fuermowei/PAMI/assets/47769416/65d4a763-936a-433d-b19f-9cd5934d85fd)

## Step 1: Model preparation
Loading a pre-trained model and adjusting it to inference mode. Meanwhile, freezing the parameter updates can effectively reduce the computation time.
```
import torchvision.models as models

model = models.vgg19_bn(models.VGG19_BN_Weights.IMAGENET1K_V1).eval()
for param in model.parameters():
    param.requires_grad = False
model = model.cuda()
```

## Step 2: Data preparation
Select data that matches the distribution used during model training and remember the corresponding data augmentation transformations (such as mean and variance normalization).
The data is in the form of a tensor when inputted into the method and has no subsequent processing (i.e. normalization).
```
from PIL import Image
from timm.data import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN

normalize = tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
transformer = tf.Compose([
        tf.Resize((224, 224)),
        tf.ToTensor()
    ])
    
url, cls = 'Image link', 'Image category number'
image = Image.open(url).convert('RGB')
image = transformer(image).unsqueeze(0).cuda()
target = torch.LongTensor([cls])
```


## Step 3: Setting hyper-parameters
Import our method and setting hyperparameters for it.
There are approximately four aspects to the settings of the method:
- Transformations needed for the image afterwards, such as normalization.
- Settings for the segmentation method, which can be referenced through external functions. 
- Masking method, such as black/white/blur as mentioned in the paper, which can be replaced by more possible methods in the future (e.g. using generative networks for filling).
- 
```

```

## Step 4: Running the method

## Step 5: Visualizing the results
