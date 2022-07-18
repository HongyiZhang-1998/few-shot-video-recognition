import os
from tqdm import tqdm
import time
from utils import getAvaliableDevice
import random

os.chdir('.')

# #dataset='Office-31'
datasets = ["UCF101"]  # ntu120_100 ntu120_30
left = False
seeds = [2021]
num_segments = [16]
dtws = ['sdtw']
py = "train.py "
vats = [0, 1]
dtw_threshold = [0]
shot = 1
alphas = [1]
archs = ['3dresnet18']

nowtime = time.strftime("%Y_%m_%d", time.localtime())

total = len(seeds) * len(vats) * len(datasets) * len(num_segments)
cnt = 0

for dataset in datasets:
    for seed in seeds:
        for num_segment in num_segments:
            for dtw in dtws:
                for vat in vats:
                    for arch in archs:
                        gpu = getAvaliableDevice(gpu=[3, 4, 5], left=left, min_mem=22000)
                        cnt += 1
                        # if cnt == 1:
                        #     continue
                        exp =  os.path.join(dataset, nowtime, dtw, 'segments_' + str(num_segment))
                        # exp = os.path.join('train_model', 'vat_{}'.format(vat))
                        command = "python " + py + " --experiment_root {} --device {} -seed {}  --dtw {} --vat {} --arch {} --num_segments {} &".format(
                        exp, gpu, seed, dtw, vat, arch, num_segment)

                        # command = "python " + py + " --experiment_root {} --device {} -seed {}  --dtw {} --vat {} --alpha {}&".format(
                        #     os.path.join(dataset, 'segments_' + str(num_segment), 'alpah_' + str(alpha)), gpu
                        #     , seed, dtw, vat, alpha)
                        print("***************************************************training on {}/{}:".format(cnt, total) + command)
                        os.system(command)

                        time.sleep(30)



