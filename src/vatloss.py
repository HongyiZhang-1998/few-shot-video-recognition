import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import gl

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d = d / (torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8)
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, y, pred_loss, model_loss=None):
        with torch.no_grad():
            # pred = F.softmax(pred_loss, dim=0)
            pred = pred_loss

        # prepare random unit tensor
        d =  (torch.rand(x.shape[1], x.shape[1]).to(x.device)) # T * T
        d = _l2_normalize(d)
        # print('d', d)

        #loss function 改成KL散度、crossentropy、分类loss
        loss_func = nn.L1Loss()

        with _disable_tracking_bn_stats(model):
        # calc adversarial direction
            for _ in range(self.ip):  #3-5
                d.requires_grad_()
                d = self.xi * d
                # print('d', d)
                d.retain_grad()

                _, _, pred_hat = model_loss(x, y, gl.shot, dtw=gl.dtw, d=d)
                # logp_hat = F.log_softmax(pred_hat, dim=0)
                # adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance = loss_func(pred_hat, pred)
                # print('adv_dis', adv_distance)
                adv_distance.backward(retain_graph=True)

                d = _l2_normalize(d.grad)
                # d = d.grad
                # print(d)
                model.zero_grad()

            # calc LDS
            # print('r_adv', d)
            r_adv = d * self.eps   #0-0.5 max(0, d-0.5)

            import os
            import numpy as np
            save_dir = '{}/r_adv'.format(gl.experiment_root)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_dir_D = '{}/D_'.format(gl.experiment_root)
            if not os.path.exists(save_dir_D):
                os.mkdir(save_dir_D)
            if gl.iter % 100 == 0 and gl.mod == 'train':
                np.savetxt(os.path.join(save_dir, 'epoch{}_iter_{}.txt'.format(gl.epoch, gl.iter)), r_adv.cpu().detach().numpy())

            # print('r_adv', r_adv)
            _, _, pred_hat = model_loss(x, y, gl.shot, dtw=gl.dtw, d=r_adv)
            # logp_hat = F.log_softmax(pred_hat, dim=0)
            # print('pred hat', pred_hat)
            # lds = F.kl_div(logp_hat, pred, reduction='batchmean')
            lds = loss_func(pred_hat, pred)
            # print('lds', lds)
        return lds
