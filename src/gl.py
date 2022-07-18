import numpy as np

epoch=0
device='cuda:5'
device_ids=[3,4]
experiment_root='../output'
debug=False
local_match=0
gamma=0.1
iter=0
R_=np.random.randn(250, 15, 15)
D_=np.random.randn(250, 15, 15)
E_=np.random.randn(250, 15, 15)
mod='train'
run_mode='train'
dataset='ntu120'
use_bias=0
n_class=5
shot=1
n_query=5
classes_iter=[]
dtw=1
eps=1
OTAM_R=np.random.randn(250, 15, 15)
OTAM_D=np.random.randn(250,15,15)
dtw_threshold=0
vat=0