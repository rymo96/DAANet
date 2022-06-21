Specific Usage
====

Preprocessing
----
Modify the original code
```
from torchvision import transforms
```
to
```
from cv2_transform import transforms
```

The names of functions and parameters remain unchanged.


Reading data
----
When reading a image, modify the default code
```
from PIL import Image  
img = Image.open(fpath)
```
to
```
from cv2_transform.functional import imread
img = imread(fpath) //default: RGB
```


Introducing RandAug
----
```
from cv2_transform.autoaug import RandAugment
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

rand_aa_policy = RandAugment(2, 11)
transform_train.transforms.insert(0, rand_aa_policy)
```

Implemented functions
----
|  Preprocessing operations   | Description  |
|  ----  | ----  |
| Resize  | Image scale transformation, support long side Resize and short side Resize, default short side |
| RandomCrop | Random crop |
| CenterCrop | Center crop |
| ToTensor | Normalization, the default value is 255, transform HWC to CHW, transform to tensor |
| Pad | Padding |
| Grayscale | Graying |
| Normalize | Divide the mean by the standard deviation |
| ColorJitter | Color jittering |
| RandomHorizontalFlip | Random horizontal flip |
| RandomVerticalFlip | Random vertical flip |
| RandomResizedCrop | Random scaling and cropping, the default configuration for training on ImageNet |
| RandomRotation | Random rotation |
| FastRotation | Rotate 90 degrees, 180 degrees, 270 degrees |
| ConditionRotation | Determine whether to rotate 90 degrees plus or minus according to the ratio of length to width |
| ColorTrans | Color transform, default is BGR to RGB |
| PadTarget | Fill the edge of the image to the specified size, including left up, left down, right up, right down, uniform five filling methods |
