# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from protonet import ProtoNet
from utils import get_para_num, setup_seed

from models import TSN

from transforms import *
from parser_video import get_parser
from VideoDataset import TSNDataSet
import numpy as np
import torch
import os
import gl
from utils import getAvaliableDevice
import tqdm

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

def get_augmentation(input_size):
    return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(is_flow=False)])


def init_dataset(args, file_list, model_name, mode):

    if 'resnet' or 'C3d' in model_name:
        crop_size = 224
        scale_size = crop_size * 256 // 224
        input_mean = [0.485, 0.456, 0.406]
        input_std = [0.229, 0.224, 0.225]
        train_augmentation = get_augmentation(crop_size)
    else :
        raise ValueError('Unknown base model: {}'.format(model_name))

    if not args.modality == 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    #取多帧的连续长度
    data_length = 1
    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['RGBDiff']: #连续多帧的差
        data_length = 5

    if mode == 'train':
        dataset = TSNDataSet(file_list,
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
    else :
        dataset = TSNDataSet(file_list,
                                 num_segments=args.num_segments,
                                 new_length=data_length,
                                 modality=args.modality,
                                 random_shift=False,
                                 test_mode=True,
                                 transform=torchvision.transforms.Compose([
                                     GroupScale(int(scale_size)),
                                     GroupCenterCrop(crop_size),
                                     Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                     ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                     normalize,
                                 ]),
                                 multi_clip_test=False,
                                 dense_sample=args.dense_sample)
    n_classes = len(np.unique(dataset.labels))
    if n_classes < args.classes_per_it_tr or n_classes < args.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset


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

def init_dataloader(opt, file_list, model_name, mode):
    dataset = init_dataset(opt, file_list, model_name, mode)
    sampler = init_sampler(opt, dataset.labels, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=opt.workers)
    return dataloader

def init_optim(opt, model, policies):
    '''
    Initialize optimizer
    '''
    if policies is not None:
        optimizer = torch.optim.SGD(policies, lr=opt.learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate, weight_decay=5e-4)

    return optimizer

def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    if opt.lr_flag == 'reduceLR':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6)
    elif opt.lr_flag == 'stepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=opt.lr_scheduler_gamma,
                                                       step_size=opt.lr_scheduler_step)

    return lr_scheduler

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None, loss_func=None):
    '''
    Train the model with the prototypical learning algorithm
    '''
    import json
    with open(os.path.join(opt.experiment_root, 'opt.json'), 'w') as f:
        j = vars(opt)
        json.dump(j, f)
        f.write('\n')

    if val_dataloader is None:
        best_state = None

    best_acc = 0
    start_epoch = 0
    patience = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')
    trace_file = os.path.join(opt.experiment_root, 'trace.txt')

    from vatloss import VATLoss
    vat_loss = VATLoss(xi=opt.xi, eps=opt.eps, ip=opt.ip)

    for epoch in range(start_epoch, opt.epochs):
        gl.epoch = epoch
        gl.iter = 0
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        train_acc = []
        train_loss = []
        lds_loss = []
        gl.iter = 0
        # for batch in tr_iter:
        for batch in (tr_iter):
            optim.zero_grad()
            gl.mod = 'train'
            gl.iter += 1
            x, y = batch
            x, y = x.to(gl.device).float(), y.to(gl.device)
            model_output = model(x)
            # print('output.size', model_output.size())
            loss, acc, batch_loss = loss_func(model_output, y, opt.num_support_tr, dtw=opt.dtw)
            # print('loss ', loss, ' acc ', acc)
            if opt.vat > 0:
                lds = vat_loss(model, model_output, y, batch_loss, model_loss=loss_func)
                loss = loss + lds * opt.alpha
                lds_loss.append((lds * opt.alpha).item())

            train_loss.append(loss.item())
            train_acc.append(acc.item())

            loss.backward()
            optim.step()

        avg_loss = np.mean(train_loss)
        avg_acc = np.mean(train_acc)
        avg_lds_loss = np.mean(lds_loss)

        if opt.vat > 0 :
            classfier_loss = avg_loss - avg_lds_loss
        else :
            classfier_loss = avg_loss

        if opt.vat > 0 :
            string = 'train loss:{}, classfier loss:{}, lds loss:{}, train Acc:{}'.format(avg_loss, classfier_loss,avg_lds_loss,avg_acc)
        else:
            string = 'train loss:{}, train Acc:{}'.format(avg_loss, avg_acc)

        if opt.lr_flag == 'reduceLR':
            lr_scheduler.step(avg_loss)
        elif opt.lr_flag == 'stepLR':
            lr_scheduler.step()

        lr = optim.state_dict()['param_groups'][0]['lr']

        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)

        model.eval()
        val_loss = []
        val_acc = []
        gl.iter = 0
        for batch in (val_iter):
            x, y = batch
            x, y = x.to(gl.device).float(), y.to(gl.device)
            gl.mod = 'val'
            gl.iter += 1
            model_output = model(x)
            # loss, acc, _ = model.loss(model_output, target=y, n_support=opt.num_support_val, dtw=opt.dtw)
            loss, acc, _ = loss_func(model_output, target=y, n_support=opt.num_support_val, dtw=opt.dtw)

            val_loss.append(loss.item())
            val_acc.append(acc.item())

        avg_loss = np.mean(val_loss)
        avg_acc = np.mean(val_acc)

        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)
        string_val = 'val loss: {}, val acc: {}{} lr:{}'.format(avg_loss, avg_acc, postfix, lr)
        print(string + '\t' + string_val)
        with open(trace_file, 'a') as f:
            f.write(string + '\t' + string_val + '\n')

        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
        if patience > 50 or classfier_loss < 0.1:
            break

    torch.save(model.state_dict(), last_model_path)

    return best_state, best_acc

def test(opt, test_dataloader, model, loss_func=None):
    '''
    Test the model trained with the prototypical learning algorithm
    '''

    avg_acc = list()
    trace_file = os.path.join(opt.experiment_root, 'test.txt')

    for epoch in range(10):
        print('=== Epoch: {} ==='.format(epoch))
        model.eval()

        gl.epoch = epoch
        gl.iter = 0
        gl.mod = 'test'

        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            # x, y = x.cuda(), y.cuda()
            x, y = x.to(gl.device).float(), y.to(gl.device)
            model_output = model(x)
            # _, acc, _ = model.loss(model_output, target=y, n_support=opt.num_support_val, dtw=opt.dtw)
            _, acc, _ = loss_func(model_output, target=y, n_support=opt.num_support_val, dtw=opt.dtw)
            avg_acc.append(acc.item())
            # print(acc)
    avg_acc = np.mean(avg_acc)
    with open(trace_file, 'a') as f:
        f.write('test acc: {}\n'.format(avg_acc))
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc

def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    options.experiment_root = os.path.join(options.experiment_root, "seed_" + str(str(options.manual_seed)), "backbone_"+str(options.arch), "vat_"+str(options.vat))
    if options.debug == 1:
        gl.debug = True
    options.cuda = True

    gl.dtw_threshold = options.dtw_threshold
    gl.gamma = options.gamma
    options.experiment_root = "../log/" + options.experiment_root
    gl.experiment_root = options.experiment_root

    gl.vat = options.vat
    gl.dataset = options.dataset
    gl.run_mode = options.mode
    gl.n_class = options.classes_per_it_tr
    gl.shot = options.num_support_tr
    gl.n_query = options.num_query_tr
    gl.dtw = options.dtw
    gl.eps = options.eps

    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    setup_seed(options.manual_seed)

    path = '/data/zhanghongyi/data/PaddleVideo/data/ucf101/split_file'

    tr_frames_path = os.path.join(path, 'tr_frames.list')
    val_frames_path = os.path.join(path, 'val_frames.list')
    test_frames_path = os.path.join(path, 'test_frames.list')

    tr_dataloader = init_dataloader(options, tr_frames_path, model_name=options.arch, mode='train')
    val_dataloader = init_dataloader(options, val_frames_path, model_name=options.arch, mode='val')
    test_dataloader = init_dataloader(options, test_frames_path, model_name=options.arch, mode='test')


    model = ProtoNet(options)
    if options.model == 1:
        model_path = os.path.join("../log", 'train_model', 'vat_{}'.format(options.vat),'best_model.pth')
        model.load_state_dict(torch.load(model_path))


    if '3d' in options.arch:
        policies = None
    else :
        policies = model.model.get_optim_policies()

    loss_func = model.loss
    print(get_para_num(model))

    model = torch.nn.DataParallel(model, device_ids=[1, 2, 3])
    # options.device = str(getAvaliableDevice(gpu=[3, 4, 5], left=False, min_mem=22000))
    device = 'cuda:{}'.format(options.device) if torch.cuda.is_available() and options.cuda else 'cpu'
    gl.device = device
    model = model.cuda(device=int(options.device))

    optim = init_optim(options, model, policies)
    lr_scheduler = init_lr_scheduler(options, optim)

    if options.mode == 'train':
        res = train(opt=options,
                    tr_dataloader=tr_dataloader,
                    val_dataloader=val_dataloader,
                    loss_func=loss_func,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler)
        best_state, best_acc = res
        print('Testing with last model..')
        test(opt=options,
             test_dataloader=test_dataloader,
             model=model,
             loss_func=loss_func)

        model.load_state_dict(best_state)
        model_path = os.path.join(options.experiment_root, 'best_model.pth')
        model.load_state_dict(torch.load(model_path))
        print('Testing with best model..')
        test(opt=options,
             test_dataloader=test_dataloader,
             model=model,
             loss_func=loss_func)
    elif options.mode == 'test':
        print('Testing with best model..')
        test(opt=options,
             test_dataloader=test_dataloader,
             model=model,
             loss_func=loss_func)


if __name__ == '__main__':
    torch.set_printoptions(
        precision=4,  # 精度，保留小数点后几位，默认4
        threshold=1000,
        edgeitems=3,
        linewidth=150,  # 每行最多显示的字符数，默认80，超过则换行显示
        profile=None,
        sci_mode=False  # 用科学技术法显示数据，默认True
    )
    main()
