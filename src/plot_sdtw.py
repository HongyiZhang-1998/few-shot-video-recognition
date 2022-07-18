from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
import os
import csv
import os
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--data_dir', type=str, default="/data/zhanghongyi/user/zhanghongyi/video-dtw/log/2022-05-09/seed_2021/backbone_3dresnet18/vat_1/")
parse.add_argument('--category', type=str, default="R_")
parse.add_argument('--epoch', type=int, default=0)
args = parse.parse_args()

data_path = os.path.join(args.data_dir, args.category)
save_path = os.path.join(args.data_dir, args.category + 'heat_map')
if not os.path.exists(save_path):
    os.mkdir(save_path)

lists = os.listdir(data_path)
lists = lists[-10:]
print(lists)
plot_epoch = args.epoch

for i in lists:
    i_num = i.split('_')[0].split('epoch')[-1]
    # if i_num != str(plot_epoch):
    #     continue
    path = os.path.join(data_path, i)
    data = np.load(path)
    n, _, _ = data.shape
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    matrix = data.sum(axis=0) / n

    if args.category == 'D_':
        matrix = matrix[0:-1, 0:-1]
    else:
        matrix = matrix[1:-1, 1:-1]

    p1 = sns.heatmap(matrix, annot=False, fmt='.0f', ax=ax)
    ax.set_title('Heat Map')
    ax.set_xlabel('')
    ax.set_ylabel('')
    s1 = p1.get_figure()

    save_file_path = os.path.join(save_path, i.split('.')[0] + '.jpg')
    matrix_path = os.path.join(save_path, i.split('.')[0] + '.csv')
    s1.savefig(save_file_path, dpi=600, bbox_inches='tight')
    np.savetxt(matrix_path, matrix, fmt='%s', delimiter=',')


path = os.path.join(data_path, lists[-1])
data = np.load(path)
n, _, _ = data.shape

for j in range(0, 20):
    if args.category == 'D_':
        matrix = data[j, 0:-1, 0:-1]
    else:
        matrix = data[j, 1:-1, 1:-1]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    p2 = sns.heatmap(matrix, annot=False, fmt='.0f', ax=ax)
    ax.set_title('Heat Map')
    ax.set_xlabel('')
    ax.set_ylabel('')
    s2 = p2.get_figure()
    single = os.path.join(save_path, 'single_' + lists[-1].split('.')[0])
    if not os.path.exists(single):
        os.mkdir(single)
    single_data_path = os.path.join(single, str(j) + '.jpg')
    s2.savefig(single_data_path, dpi=600, bbox_inches='tight')

    matrix_path = os.path.join(single, str(j) + '.csv')
    np.savetxt(matrix_path, matrix, fmt='%s', delimiter=',')