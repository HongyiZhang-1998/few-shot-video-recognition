# import os
# from transforms import *
#
# def _load_image(directory, idx):
#     image_tmpl = 'img_{:05d}.jpg'
#     try:
#         return [Image.open(os.path.join(directory, image_tmpl.format(idx))).convert('RGB')]
#     except Exception:
#         print('error loading image:', os.path.join(directory, image_tmpl.format(idx)))
#         return [Image.open(os.path.join(directory, image_tmpl.format(1))).convert('RGB')]
#
# directory = '/mnt/data5/PaddleVideo/data/ucf101/rawframes/BabyCrawling/v_BabyCrawling_g15_c02'
#
# x = _load_image(directory, 10)
#
# print(x)
#
# normalize = IdentityTransform()
#
#
# def get_augmentation():
#     input_size = 224
#     return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
#                                                GroupRandomHorizontalFlip(is_flow=False)])
#
# train_augmentation = get_augmentation()
# transforms=torchvision.transforms.Compose([
#     train_augmentation,
#                        Stack(roll=('resnet101' in ['BNInception', 'InceptionV3'])),
#                        ToTorchFormatTensor(div=('resnet101' not in ['BNInception', 'InceptionV3'])),
#                        normalize,
#                    ])
#
# x = transforms(x)
#
# print(x)
# print(x.shape)
# from torch import nn
# from torchvision import datasets, transforms, models
# model_ft = models.resnet101(pretrained=True)
# print('1*********************************************************')
# print(model_ft)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 5)
# print('2*********************************************************')
# print(model_ft)
#
# model_ft=models.resnet101(pretrained=True)
# del model_ft.fc
# model_ft.fc=lambda x:x
# print('3*********************************************************')
# print(model_ft)
import os

import numpy as np
file = "/mnt/data4/val_k400/val_rawframes"
classes = os.listdir(file)
classes.sort()
save_file = "/mnt/data4/val_k400/split_file/frames.list"
# file_string = []
# for label, class_name in enumerate(classes):
#     class_path = os.path.join(file, class_name)
#     video_list = os.listdir(class_path)
#     for video in video_list:
#         frame_list_path = os.path.join(class_path, video)
#         frame_list = os.listdir(frame_list_path)
#         length = len(frame_list)
#         strs = (frame_list_path + " " + str(length) + " " + str(label) +"\n")
#         file_string.append(strs)
#
# fp = open(save_file, 'w')
# for s in file_string:
#     fp.write(s)
# fp.close()
# dict = {}
# import random
# classes = [x for x in range(101)]
# random.shuffle(classes)
# classes = np.array(classes)
#
# tr_classes = classes[:80]
# val_classes = classes[80:90]
# test_classes = classes[90:]
# np.savetxt('/mnt/data4/PaddleVideo/data/ucf101/split_file/tr_classes.txt', tr_classes, fmt='%d')
# np.savetxt('/mnt/data4/PaddleVideo/data/ucf101/split_file/val_classes.txt', val_classes, fmt='%d')
# np.savetxt('/mnt/data4/PaddleVideo/data/ucf101/split_file/test_classes.txt', test_classes, fmt='%d')

#
tr_classes = np.loadtxt('/mnt/data4/val_k400/split_file/tr_classes.txt').astype(np.int)
val_classes = np.loadtxt('/mnt/data4/val_k400/split_file/val_classes.txt').astype(np.int)
test_classes = np.loadtxt('/mnt/data4/val_k400/split_file/test_classes.txt').astype(np.int)
#
#
# print(val_classes, '\n', test_classes)
tr_frames_path = '/mnt/data4/val_k400/split_file/tr_frames.list'
val_frames_path = '/mnt/data4/val_k400/split_file/val_frames.list'
test_frames_path = '/mnt/data4/val_k400/split_file/test_frames.list'

# val = np.load(val_frames_path + '.npy')
# print(len(val))

tr_frames, val_frames, test_frames = [], [], []
# save_tr_file = '/mnt/data4/PaddleVideo/data/ucf101/split_file/ucf101_train_split_1_rawframes.txt'
# save_test_file = '/mnt/data4/PaddleVideo/data/ucf101/split_file/ucf101_val_split_1_rawframes.txt'
with open(save_file, 'r') as f:
    lines = f.read()
    line_list = lines.split("\n")
    line_list = line_list[:-2]
    for line in line_list:
        new_line = '/mnt/data4/val_k400/val_rawframes' + line
        lists = line.split(" ")
        if int(lists[2]) in tr_classes:
            tr_frames.append(new_line)
        elif int(lists[2]) in val_classes:
            val_frames.append(new_line)
        else :
            test_frames.append(new_line)

tr_frames = np.array(tr_frames)
val_frames = np.array(val_frames)
test_frames = np.array(test_frames)
#
# tmp = [x.strip().split(' ') for x in open(val_frames_path)]
# print(tmp)

with open(tr_frames_path, 'a') as f:
    for line in tr_frames:
        f.write(line + '\n')
with open(val_frames_path, 'a') as f:
    for line in val_frames:
        f.write(line + '\n')
with open(test_frames_path, 'a') as f:
    for line in test_frames:
        f.write(line + '\n')

# np.savetxt(tr_frames_path, tr_frames)
# np.savetxt(val_frames_path, val_frames)
# np.savetxt(test_frames_path, test_frames)




