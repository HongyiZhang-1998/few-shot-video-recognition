import numpy as np
import os

import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import torch
from tsn import TSN
from VideoDataset import TSNDataSet

from transforms import *
import dataset_config
from parser_video import get_parser
from batch_sampler import PrototypicalBatchSampler

args = get_parser().parse_args()
def get_para_num(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

dataset = 'ucf101'
num_class = 10
model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)



crop_size = model.crop_size
scale_size = model.scale_size
input_mean = model.input_mean
input_std = model.input_std
policies = model.get_optim_policies()
train_augmentation = model.get_augmentation()

if not args.modality == 'RGBDiff':
    normalize = GroupNormalize(input_mean, input_std)
else:
    normalize = IdentityTransform()

if args.modality == 'RGB':
    data_length = 1
elif args.modality in ['RGBDiff']:
    data_length = 5

train_list = '/mnt/data5/PaddleVideo/data/ucf101/split_file/ucf101_train_split_1_rawframes.txt'
val_list = '/mnt/data5/PaddleVideo/data/ucf101/split_file/ucf101_val_split_1_rawframes.txt'

train_dataset = TSNDataSet(train_list,
                   num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]),
                   multi_clip_test=True,
                   dense_sample=args.dense_sample)
val_dataset = TSNDataSet(val_list,
               num_segments=args.num_segments,
               new_length=data_length,
               modality=args.modality,
               random_shift=False,
               transform=torchvision.transforms.Compose([
                   GroupScale(int(scale_size)),
                   GroupCenterCrop(crop_size),
                   Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                   ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                   normalize,
               ]),
               multi_clip_test=False,
               dense_sample=args.dense_sample)


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
        iters = opt.train_iterations
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val
        iters = opt.test_iterations

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=iters)
# print(train_dataset.labels)
sampler = init_sampler(args, train_dataset.labels, 'train')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler, num_workers=args.workers)


# train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.workers,
#         pin_memory=True,
#         drop_last=True)


tr_iter = iter(train_loader)
for batch in (tr_iter):
    x, y = batch
    print('x.shape ', x.shape)
    print('batch labels', y)
    break

print(get_para_num(model))


