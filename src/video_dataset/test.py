
from PIL import Image
import os
from transforms import *

def _load_image(directory, idx):
    image_tmpl = 'img_{:05d}.jpg'
    try:
        return [Image.open(os.path.join(directory, image_tmpl.format(idx))).convert('RGB')]
    except Exception:
        print('error loading image:', os.path.join(directory, self.image_tmpl.format(idx)))
        return [Image.open(os.path.join(directory, image_tmpl.format(1))).convert('RGB')]

directory = '/mnt/data5/PaddleVideo/data/ucf101/rawframes/BabyCrawling/v_BabyCrawling_g15_c02'

x = _load_image(directory, 10)

print(x)

normalize = IdentityTransform()


def get_augmentation():
    input_size = 224
    return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(is_flow=False)])

train_augmentation = get_augmentation()
transforms=torchvision.transforms.Compose([
    train_augmentation,
                       Stack(roll=('resnet101' in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=('resnet101' not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ])

x = transforms(x)

print(x)
print(x.shape)