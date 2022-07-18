import numpy as np
import torch
from numba import jit
from torch.autograd import Function
from torch.nn import functional as F
import gl
import os
# from utils import plot_matrix
from cross_attention import PositionEncoding

@jit(nopython = True)
def compute_softdtw(D, gamma):
  B = D.shape[0]
  N = D.shape[1]
  M = D.shape[2]
  R = np.ones((B, N + 2, M + 2)) * np.inf
  R[:, 0, 0] = 0
  for k in range(B):
    for j in range(1, M + 1):
      for i in range(1, N + 1):
        if gl.dtw_threshold > 0 and abs(i - j) >= float(gl.dtw_threshold) :
          continue
        r0 = -R[k, i - 1, j - 1] / gamma
        r1 = -R[k, i - 1, j] / gamma
        r2 = -R[k, i, j - 1] / gamma
        rmax = max(max(r0, r1), r2)
        rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
        softmin = - gamma * (np.log(rsum) + rmax)
        R[k, i, j] = D[k, i - 1, j - 1] + softmin
  # print('*********************DDDDDDDDDDDDDDDDDDDDDDDDDDD**********************')
  # print(D[0])
  # print('***************RRRRRRRRRRRRRRRRRRRRRRRRR******************')
  # print(R[0, 1:N + 1, 1:M + 1])
  return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
  B = D_.shape[0]
  N = D_.shape[1]
  M = D_.shape[2]
  D = np.zeros((B, N + 2, M + 2))
  E = np.zeros((B, N + 2, M + 2))
  D[:, 1:N + 1, 1:M + 1] = D_
  E[:, -1, -1] = 1
  R[:, : , -1] = -np.inf
  R[:, -1, :] = -np.inf
  R[:, -1, -1] = R[:, -2, -2]
  for k in range(B):
    for j in range(M, 0, -1):
      for i in range(N, 0, -1):
        if gl.dtw_threshold > 0 and abs(i - j) >= float(gl.dtw_threshold) :
          continue
        a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
        b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
        c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
        a = np.exp(a0)
        b = np.exp(b0)
        c = np.exp(c0)
        E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
  # print('***************EEEEEEEEEEEEEEEEEE******************')
  # print(E[0, 1:N + 1, 1:M + 1])
  return E[:, 1:N + 1, 1:M + 1]

class _SoftDTW(Function):
  @staticmethod
  def forward(ctx, D, gamma):
    dev = D.device
    dtype = D.dtype
    gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
    D_ = D.detach().cpu().numpy()
    gl.D_ = D_
    g_ = gamma.item()
    R = torch.Tensor(compute_softdtw(D_, g_)).to(dev).type(dtype)
    gl.R_ = R.detach().cpu().numpy()

    ctx.save_for_backward(D, R, gamma)
    return R[:, -2, -2]

  @staticmethod
  def backward(ctx, grad_output):
    dev = grad_output.device
    dtype = grad_output.dtype
    D, R, gamma = ctx.saved_tensors
    D_ = D.detach().cpu().numpy()
    R_ = R.detach().cpu().numpy()
    g_ = gamma.item()
    E = torch.Tensor(compute_softdtw_backward(D_, R_, g_)).to(dev).type(dtype)
    return grad_output.view(-1, 1, 1).expand_as(E) * E, None

class SoftDTW(torch.nn.Module):
  def __init__(self, gamma=1.0, normalize=False, attention=None, attention_y=None, pos_enc=None, bias=None, d=None):
    super(SoftDTW, self).__init__()
    self.normalize = normalize
    self.gamma = gamma
    self.func_dtw = _SoftDTW.apply
    self.attention = attention
    self.attention_y = attention_y
    self.pos_enc = pos_enc
    self.bias = bias
    self.d = d

    if attention != None:
      self.calc_matrix_func = self.attention_calc_distance_matrix

    else:
      self.calc_matrix_func = self.calc_distance_matrix

  def attention_calc_distance_matrix(self, x, y):
    n, t, v, c = x.size()

    x = x.view(n * t, v, c)
    y = y.view(n * t, v, c)

    # x = x + self.pos_enc
    # y = y + self.pos_enc

    attention_x = self.attention(x, y, debug=False)
    attention_y = self.attention_y(y, x, debug=True)

    # if gl.epoch % 5 == 0 and gl.iter % 100 == 0:
    #   path = os.path.join(gl.experiment_root, 'save_attention_xy', 'epoch_{}_iter_{}'.format(gl.epoch, gl.iter))
    #   if not os.path.exists(path):
    #     os.makedirs(path)
    #   np.save(os.path.join(path, 'attention_x.npy'), attention_x.cpu().detach().numpy())
    #   np.save(os.path.join(path, 'attention_y.npy'), attention_y.cpu().detach().numpy())

    attention_x = attention_x.view(n, t, -1)
    attention_y = attention_y.view(n, t, -1)

    return self.calc_distance_matrix(attention_x, attention_y)

  def calc_distance_matrix(self, x, y):
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)

    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)

    e_cos = torch.cosine_similarity(x, y, dim=3)

    # x = x.reshape(-1, d)
    # y = y.reshape(-1, d)
    # x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    # y = y / (y.norm(dim=1, keepdim=True) + 1e-8)
    #
    # cos = x * y
    # e_cos = cos.sum(1)
    # e_cos = e_cos.view(-1, n, m)

    #add random nosiy
    if gl.mod == 'train' and self.d != None:
      # print('self.d', self.d.size())
      e_cos =  e_cos + self.d

    return 1 - e_cos  # 优化1-ecos小，e_cos大, 越相似

  def forward(self, x, y):
    assert len(x.shape) == len(y.shape)
    squeeze = False
    if len(x.shape) < 3:
      x = x.unsqueeze(0)
      y = y.unsqueeze(0)
      squeeze = True
    if self.normalize:
      D_xx = self.calc_matrix_func(x, x)
      out_xx = self.func_dtw(D_xx, self.gamma)
      D_yy = self.calc_matrix_func(y, y)
      out_yy = self.func_dtw(D_yy, self.gamma)
      D_xy = self.calc_matrix_func(x, y)
      out_xy = self.func_dtw(D_xy, self.gamma)

      result = out_xy - 1/2 * (out_xx + out_yy)  # distance
    else:
      D_xy = self.calc_matrix_func(x, y)
      out_xy = self.func_dtw(D_xy, self.gamma)
      result = out_xy  # discrepancy

    import os
    save_dir = '{}/R_'.format(gl.experiment_root)
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)
    save_dir_D = '{}/D_'.format(gl.experiment_root)
    if not os.path.exists(save_dir_D):
      os.mkdir(save_dir_D)
    if gl.iter % 100 == 0:
      np.save(os.path.join(save_dir, 'epoch{}_iter_{}.npy'.format(gl.epoch, gl.iter)), gl.R_)
      np.save(os.path.join(save_dir_D, 'epoch{}_iter_{}.npy'.format(gl.epoch, gl.iter)), gl.D_)

    return result.squeeze(0) if squeeze else result
