具体用法
====

预处理
----
将代码中原有的
```
from torchvision import transforms
```
修改为
```
from cv2_transform import transforms
```
即可，函数名和参数名保持一致，并未有变化。

数据读取
----
读取图片时，将默认的
```
from PIL import Image  
img = Image.open(fpath)
```
修改为
```
from cv2_transform.functional import imread
img = imread(fpath) //默认为RGB
```

引入RandAug
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

已实现函数
----
|  预处理操作   | 功能描述  |
|  ----  | ----  |
| Resize  | 图像尺度变换，支持长边Resize和短边Resize，默认为短边 |
| RandomCrop | 随机裁切 |
| CenterCrop | 中心裁切 |
| ToTensor | 归一化，默认值为255，hwc变为chw，转为tensor |
| Pad | 补边填充 |
| Grayscale | 灰度化 |
| Normalize | 去均值除标准差 |
| ColorJitter | 色彩抖动 |
| RandomHorizontalFlip | 随机水平翻转 |
| RandomVerticalFlip | 随机竖直翻转 |
| RandomResizedCrop | 随机放缩裁切，imagenet训练默认配置 |
| RandomRotation | 随机旋转 |
| FastRotation | 旋转90度，180度，270度 |
| ConditionRotation | 根据长宽比例判断是否需要正负旋转90度 |
| ColorTrans | 色彩变换，默认为BGR to RGB |
| PadTarget | 将图像补边至指定尺寸，包含左上、左下、右上、右下、均匀五种补边方式
