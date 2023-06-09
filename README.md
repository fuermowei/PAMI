# PAMI: Partition input and Aggregate outputs for Model Interpretation
This is the repository for Pytorch Implementation of "PAMI: Partition input and Aggregate outputs for Model Interpretation". You are welcome to raise any questions or concerns about this repository in this issue, and the authors will respond as soon as possible.

## Modules
In addition to the main branch, there are four other branches in the repository.
- **functional**: It contains some useful functions such as visualization, smoothing, and other operations.
- **methods**: They also contain the code for the methods used in the comparative experiments in our paper.
- **experiment**: It contains the code for experiments such as pointing game and insertion game.
- **images**: The original images used for demonstration in the paper are provided here for users' convenience.

## Requirements
- Install pytorch, torchvision
- Install skimage, cv2, scipy, PIL
- Install timm (optional)

## Method
![MainMap](https://github.com/fuermowei/PAMI/assets/47769416/65d4a763-936a-433d-b19f-9cd5934d85fd)
The above figure is the framework of our method. For specific method details, please refer to our submitted responses or manuscript. Earlier versions can be found at https://arxiv.org/abs/2302.03318.

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
Import our method and setting hyper-parameters for it.
There are approximately four aspects to the settings of the method:
- Transformations needed for the image afterwards, such as normalization.
- Settings for the segmentation method, which can be referenced through external functions. You can use PAMI.segmentation.append((method, {hyper-parameters})) to add your segmentation method, as long as the method can return a superpixel segmentation label map with the same dimensions as the input image and numbering starting from 1. For example, PAMI.segmentation.append(felzenszwalb, {'scale': 250, 'sigma': 0.8, 'min_size': 28 * 28})).
- Masking method, such as black/white/blur as mentioned in the paper, which can be replaced by more possible methods in the future (e.g. using generative networks for filling). It is controlled through the substrate parameter, and we provide corresponding functions in the file as options.
- We recommend using a weakly augmented version of the input image that was used during model training (e.g. translation, cropping, rotation, smoothing) to generate more diverse superpixel regions. Therefore, we have implemented functions related to data augmentation to further assist in achieving this step. These functions can also be defined by the user, and they need to implement a forward function (corresponding to the function of data augmentation) and a backward function (which can match the obtained heatmap to the corresponding position in the original image).
```
from Augmentation.Identity import Identity
from Augmentation.Shift import Shift
from Aug_BatchRun import Ours
from Aug_Batchrun_Twostage import Ours as Ours2
augumentation = [Identity(),
                 Shift((36, 0)), Shift((0, 36)), Shift((-36, 0)), Shift((0, -36)),
                 Shift((30, 0)), Shift((0, 30)), Shift((-30, 0)), Shift((0, -30)),
                 Shift((24, 0)), Shift((0, 24)), Shift((-24, 0)), Shift((0, -24))]
         
from functional import felzenszwalb, slic, SEEDS, watershed_sobel, blur
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

blur_f = blur(49, 100)
ours = PAMI(transformer=tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            substrate=blur,
            segmentation=segmentation,
            augmentation=augumentation, mini_batch=90)
```

## Step 4: Running the method
The method can be run once or multiple times.
During runtime, in addition to the model, image, and label, we also provide a heatmap (optional). If a heatmap is provided, the segmentation algorithm will use the heatmap as input for segmentation instead of the image.

```
heatmap = ours(model, image, target) # for run once
heatmap = ours(model, image, target, heatmap) # for run multiple times
```

## Step 5: Visualizing the results
The output results need to be visualized, and we provide two visualization options. It's worth noting that there is not only one visualization method, for example, color enhancement or stretching within a certain range can also be used.
- Average visualization, since our method uses predicted probabilities, the average contribution of each pixel to the model's output probability is between 0 and 1. If we directly map probability 0 to 0 pixel intensity and probability 1 to 255 pixel intensity, this approach can well reflect the absolute importance of pixels to the model.
- Normalization visualization, by converting the image to Y = (X - min(X)) / (max(X) - min(X) + delta), where delta is a small positive constant to avoid division by zero. This approach can better display the relative importance within the image.
```
heatmap = heatmap / (N + 1e-9) # Average visualization
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9) # Normalization visualization
```
