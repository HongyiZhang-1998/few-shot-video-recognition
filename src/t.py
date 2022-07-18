import torch.nn.parallel
import torch.optim

from prototypical_batch_sampler import PrototypicalBatchSampler

from models import TSN
import warnings
warnings.filterwarnings("ignore")

from transforms import *
from parser_video import get_parser
from VideoDataset import TSNDataSet
import numpy as np
import torch
from basic_ops import ConsensusModule
from torchvision import datasets, transforms, models
from PIL import Image

import os
import numpy as np
from transforms import *
from parser_video import get_parser

args = get_parser().parse_args()
num_class = 512
model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

def load_image(directory, idx):
    image_tmpl = 'img_{:05d}.jpg'
    return [Image.open(os.path.join(directory, image_tmpl.format(idx))).convert('RGB')]

def get_augmentation(input_size):
    return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(is_flow=False)])

def save_file(path, mode, args, save_path):

    tmp = [x.strip().split(' ') for x in open(path)]

    crop_size = 224
    scale_size = crop_size * 256 // 224
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    train_augmentation = get_augmentation(crop_size)
    normalize = IdentityTransform()

    if 'tr' in mode:
        transform=torchvision.transforms.Compose([
                                       train_augmentation,
                                       Stack(roll=('resnet50' in ['BNInception', 'InceptionV3'])),
                                       ToTorchFormatTensor(div=('resnet50' not in ['BNInception', 'InceptionV3'])),
                                       normalize,
                                   ])
    else :
        transform = torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            GroupCenterCrop(crop_size),
            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
            normalize,
        ])

    datas = []
    labels = []
    frames =[]

    for images in tmp:
        file_path = images[0]
        frame = int(images[1])
        label = int(images[2])
        labels.append(label)
        frames.append(frame)
        image = []
        for i in range(1, frame + 1):
            process_data = np.expand_dims(transform(load_image(file_path, i)), axis=0)
            # print(process_data.shape)
            image.append(process_data)
        image = np.concatenate(image)
        image = np.expand_dims(image, axis=0)
        print(image.shape)
        datas.append(image)

    datas = np.concatenate(datas)
    labels = np.array(labels)
    frames = np.array(frames)

    print('data.shape, labels.shape, frames.shape', datas.shape, labels.shape, frames.shape)

    # np.save(os.path.join(save_path, mode + '_data.npy'), datas)
    # np.save(os.path.join(save_path, mode + '_label.npy'), labels)
    # np.save(os.path.join(save_path, mode + '_frames.npy'), frames)

save_path = "/mnt/data4/PaddleVideo/data/ucf101/Transform_ResNet"
file_path = "/mnt/data4/PaddleVideo/data/ucf101/split_file"
modes = ['tr', 'val', 'test']

for mode in modes :
    save_file(path=os.path.join(file_path, mode + '_frames.list'), mode=mode, args=args, save_path=save_path)
