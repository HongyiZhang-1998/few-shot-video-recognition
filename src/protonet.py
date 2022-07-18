import os
import time

import torch.nn as nn
import torch
import numpy as np
from utils import get_support_query_data, extract_k_segement, compute_similarity, euclidean_dist, euclidean_distance
from torch.nn import functional as F
import gl
from soft_dtw import SoftDTW
from otam.softerdtw_padquery_update2_right import OTAM_SoftDTW
from cross_attention import CrossAttention, PositionEncoding
from models import TSN
from self_attention import SelfAttention
from utils import plot_matrix
import os


class ProtoNet(nn.Module):
    def __init__(self, args):
        super(ProtoNet, self).__init__()
        num_class = args.output_features #num_class is no thing
        self.model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)


    def loss(self, input, target, n_support, dtw, d=None):
        # input is encoder by Net
        # print(input.size())
        n, t, c = input.size()

        def supp_idxs(cc):
            # FIXME when torch will support where as np
            return torch.nonzero(target.eq(cc), as_tuple=False)[:n_support].squeeze(1)\

        # FIXME when torch.unique will be available on cuda too
        classes = torch.unique(target)
        n_class = len(classes)

        # FIXME when torch will support where as np
        # assuming n_query, n_target constants
        n_query = target.eq(classes[0].item()).sum().item() - n_support
        support_idxs = list(map(supp_idxs, classes))

        z_proto = torch.stack([input[idx_list] for idx_list in support_idxs]).view(-1, t, c)
        # FIXME when torch will support where as np


        query_idxs = torch.stack(list(map(lambda c: torch.nonzero(target.eq(c), as_tuple=False)[n_support:], classes))).view(-1)
        zq = input[query_idxs.long()]
        z_proto = z_proto.view(n_class, n_support, t, c).mean(1)  # n, t, c

        dist = self.dtw_dist(zq, z_proto, d, dtw)

        log_p_y = F.log_softmax(-dist, dim=1).view(n_class, n_query, -1)

        target_inds = torch.arange(0, n_class).to(zq.device)
        # target_inds = torch.arange(0, n_class).to(gl.device)
        target_inds = target_inds.view(n_class, 1, 1)
        target_inds = target_inds.expand(n_class, n_query, 1).long()

        batch_loss = -log_p_y.gather(2, target_inds).squeeze().view(-1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)

        acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

        return loss_val, acc_val, batch_loss

    def dtw_dist(self, x, y, d, dtw):
        '''
            :param x: [n, t, c] z_query
            :param y: [m, t, c] z_proto
            :return: [n, m]
        '''

        if len(x.size()) == 4:
            n, t, v, c = x.size()
            x = x.view(n, t, v * c)
            y = y.view(-1, t, v * c)

        n, t, c = x.size()
        m, _, _ = y.size()

        x = x.unsqueeze(1).expand(n, m, t, c).reshape(n * m, t, c)
        y = y.unsqueeze(0).expand(n, m, t, c).reshape(n * m, t, c)


        if 'sdtw' in dtw:
            sdtw = SoftDTW(gamma=gl.gamma, normalize=True, d=d)
        elif 'otam' in dtw:
            sdtw = OTAM_SoftDTW(gamma=gl.gamma, use_cuda=False, d=d)
        else :
            sdtw = None

        loss = sdtw(x, y)

        return loss.view(n, m)


    def forward(self, x):
        x = self.model(x)

        return x

