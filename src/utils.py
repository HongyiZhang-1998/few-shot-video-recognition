import pickle
import fcntl
import torch
import time
import random
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
import os
import csv
import argparse
import pynvml, time

def getAvaliableDevice(gpu=[0, 1, 2, 3, 4, 5], min_mem=24000, left=False):
    """
    :param gpu:
    :param min_mem:
    :param left:
    :return:
    """

    pynvml.nvmlInit()
    t = int(time.strftime("%H", time.localtime()))

    if t >= 23 or t < 8:
        left = False  # do not leave any GPUs
    # else:
    # left=True

    min_num = 3
    dic = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, -1: -1}
    ava_gpu = -1

    while ava_gpu == -1:
        avaliable = []
        for i in gpu:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if (meminfo.free / 1024 ** 2) > min_mem:
                avaliable.append(dic[i])

        if len(avaliable) == 0 or (left and len(avaliable) <= 1):
            # if len(avaliable)==1:
            #     if avaliable[0] not in [4,5,6]:
            #         ava_gpu= -1
            #         time.sleep(5)
            #         continue
            # else :
            ava_gpu = -1
            time.sleep(20)
            continue
        ava_gpu = avaliable[0]
    print('ava_gpu:', ava_gpu)
    return ava_gpu

def plot_matrix(save_path, matrix):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    p = sns.heatmap(matrix, annot=False, fmt='.0f', ax=ax)
    ax.set_title('heat map')
    ax.set_xlabel('')
    ax.set_ylabel('')
    s = p.get_figure()
    s.savefig(save_path, dpi=600, bbox_inches='tight')

    #save_cvs_path = os.path.join(save_path.split('/')[-1].split('.jpg')[0], 'csv_file')
    save_csv_path = save_path.split('.jpg')[0] + '.csv'

    np.savetxt(save_csv_path, matrix, fmt='%s', delimiter=',')

def write_shared_file(file_name,content):
    nowtime=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    content[0]=nowtime+" "+content[0]
    with open(file_name,'a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.writelines(content)
        fcntl.flock(f,fcntl.LOCK_UN)

def write_csv_file(file_name,content):
    nowtime=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    content["time"]=nowtime
    to_write_head = False
    if not os.path.exists(file_name):
        to_write_head=True
    with open(file_name,'a+') as f:
        writer=csv.DictWriter(f,content.keys())
        fcntl.flock(f,fcntl.LOCK_EX)
        if to_write_head:
            writer.writeheader()
        writer.writerow(content)
        # for key, value in content.items:
        #     writer.writerow([key, value])
        fcntl.flock(f,fcntl.LOCK_UN)
def get_para_num(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # np.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


#欧几里得距离
def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def get_support_query_data(support, query, device):
    '''
    :param support:[n_class, c, v]
    :param query: [n_class * n_query, c, v]
    :return: sq: [n_class * (n_class * n_query) * 2, c, v]
    '''
    n_class, c, v = support.size()
    all_query = query.size(0)
    sum_matching_graph = n_class * all_query * 2

    node_features = torch.zeros(sum_matching_graph, c, v).to(device)

    idx, idx2= torch.arange(0, sum_matching_graph, 2).to(device), torch.arange(1, sum_matching_graph, 2).to(device)
    node_features[idx] = query.unsqueeze(1).repeat(1, n_class, 1, 1).reshape(-1, c, v)
    node_features[idx2] = support.unsqueeze(0).repeat(all_query, 1, 1, 1).reshape(-1, c, v)

    node_features = node_features.permute(0, 2, 1).reshape(sum_matching_graph * v, c)

    return node_features


def euclidean_distance(x, y):
    """This is the squared Euclidean distance."""
    return torch.sum((x - y) ** 2, dim=-1)

def compute_similarity(x, y):
    """Compute the distance between x and y vectors.

    The distance will be computed based on the training loss type.

    Args:
      config: a config dict.
      x: [n_examples, feature_dim] float tensor.
      y: [n_examples, feature_dim] float tensor.

    Returns:
      dist: [n_examples] float tensor.

    Raises:
      ValueError: if loss type is not supported.
    """

    return -euclidean_distance(x, y)

def extract_k_segement(x, num_frame, segement):
    n, c, t, v = x.size()

    assert n == len(num_frame)
    step = num_frame // segement

    new_x = []
    for i in range(n):
        idx = [ random.randint(j * step[i], (j + 1) * step[i] - 1) for j in range(segement)]
        new_x.append(x[i, :, idx, :].unsqueeze(0))

    new_x = torch.cat(new_x, dim=0)

    return new_x

def load_data(path, train_class_name, val_class_name, test_class_name, num):
    data_path = os.path.join(path, 'train_data_joint.npy')
    label_path = os.path.join(path, 'train_label.pkl')
    num_frame_path = os.path.join(path, 'train_num_frame.npy')
    num_class = np.zeros(125)

    try:
        with open(label_path) as f:
            sample_name, label = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
            sample_name, label = pickle.load(f, encoding='latin1')

    # load data
    data = np.load(data_path)
    num_frame = np.load(num_frame_path)
    # num_frame = np.ones(len(label)) * 300

    train_data, val_data, test_data = [], [], []
    train_label, val_label, test_label = [], [], []
    train_num_frame, val_num_frame, test_num_frame = [], [], []
    for i in range(len(label)):
        if label[i] > 121:
            continue

        num_class[label[i]] += 1

        if label[i] in train_class_name:
            if num_class[label[i]] > num:
                continue
            train_data.append(np.expand_dims(data[i], axis=0))
            train_label.append(label[i])
            train_num_frame.append(num_frame[i])
        elif label[i] in val_class_name:
            if num_class[label[i]] > num:
                continue
            val_data.append(np.expand_dims(data[i], axis=0))
            val_label.append(label[i])
            val_num_frame.append(num_frame[i])
        elif label[i] in test_class_name:
            if num_class[label[i]] > num:
                continue
            test_data.append(np.expand_dims(data[i], axis=0))
            test_label.append(label[i])
            test_num_frame.append(num_frame[i])
    train_data, val_data, test_data = np.concatenate(train_data, 0), np.concatenate(val_data, 0), np.concatenate(test_data, 0)

    save_path = '/mnt/data1/nturgbd_raw/ntu120/random_data/train_val_test_{}'.format(num)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, 'train_data.npy'), train_data)
    np.save(os.path.join(save_path, 'train_label.npy'), train_label)
    np.save(os.path.join(save_path, 'train_frame.npy'), train_num_frame)
    np.save(os.path.join(save_path, 'val_data.npy'), val_data)
    np.save(os.path.join(save_path, 'val_label.npy'), val_label)
    np.save(os.path.join(save_path, 'val_frame.npy'), val_num_frame)
    np.save(os.path.join(save_path, 'test_data.npy'), test_data)
    np.save(os.path.join(save_path, 'test_label.npy'), test_label)
    np.save(os.path.join(save_path, 'test_frame.npy'), test_num_frame)

    data_list = [train_data, train_label, np.array(train_num_frame), val_data, val_label, np.array(val_num_frame), test_data, test_label, np.array(test_num_frame)]

    return data_list

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    data_path = '/mnt/data1/nturgbd_raw/ntu120/xsub'

    path = '/mnt/data1/nturgbd_raw/ntu120/random_data/train_val_test_30'

    tr_path = os.path.join(path, 'tr_label.txt')
    tr = np.loadtxt(tr_path).astype(int)
    val_path = os.path.join(path, 'val_label.txt')
    val = np.loadtxt(val_path).astype(int)
    test_path = os.path.join(path, 'test_label.txt')
    test = np.loadtxt(test_path).astype(int)
    print(tr)
    print(val)
    print(test)

    train_class_name, val_class_name, test_class_name = tr, val, test
    data_list = load_data(data_path, train_class_name, val_class_name, test_class_name, num=60)
