import os
from tqdm import tqdm
import time
from utils import getAvaliableDevice
import random

os.chdir('.')

# #dataset='Office-31'
datasets = ["ntu120_30"]  # ntu120_100 ntu120_30

left = False

seeds = [2021]

backbones = ["st_gcn"]


py = "train.py "
lr= 0.001
shot = 1

epss = [1]
xis = [10]
alphas = [1, 0.1, 10]
ips = [3, 5]

# nowtime="2021_08_28"+dataset
total = len(backbones) * len(seeds) * len(datasets) * len(xis) * len(alphas) * len(epss) * len(ips)
cnt = 0

for backbone in backbones:
    for dataset in datasets:
        for seed in seeds:
            for alpha in alphas:
                for ip in ips:
                    for xi in xis:
                         for eps in epss:
                                if alpha == 1 and ip == 1:
                                    cnt += 1
                                    continue
                                gpu = getAvaliableDevice(gpu=[3, 4, 5], left=left, min_mem=24000)
                                cnt += 1
                                command = "python " + py + "--backbone {} --experiment_root {} --device {} -seed {} -lr {} --alpha {} --xi {} --eps {} --ip {} --dataset {} &".format(
                                        backbone, os.path.join("tune", 'alpha_{}'.format(alpha), 'ip_{}'.format(ip), 'xi_{}'.format(xi),'eps_{}'.format(eps)), gpu
                                        , seed, lr, alpha, xi, eps, ip, dataset)
                                print("***************************************************training on {}/{}:".format(cnt, total) + command)
                                os.system(command)

                                time.sleep(30)



