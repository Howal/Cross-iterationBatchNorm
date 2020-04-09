from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class CBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True,
                 buffer_num=0, rho=1.0,
                 burnin=0, two_stage=True,
                 FROZEN=False, out_p=False):
        super(CBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.buffer_num = buffer_num
        self.max_buffer_num = buffer_num
        self.rho = rho
        self.burnin = burnin
        self.two_stage = two_stage
        self.FROZEN = FROZEN
        self.out_p = out_p

        self.iter_count = 0
        self.pre_mu = []
        self.pre_meanx2 = []  # mean(x^2)
        self.pre_dmudw = []
        self.pre_dmeanx2dw = []
        self.pre_weight = []
        self.ones = torch.ones(self.num_features).cuda()

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def _update_buffer_num(self):
        if self.two_stage:
            if self.iter_count > self.burnin:
                self.buffer_num = self.max_buffer_num
            else:
                self.buffer_num = 0
        else:
            self.buffer_num = int(self.max_buffer_num * min(self.iter_count / self.burnin, 1.0))

    def forward(self, input, weight):
        # deal with wight and grad of self.pre_dxdw!
        self._check_input_dim(input)
        y = input.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(input.size(1), -1)

        # burnin
        if self.training and self.burnin > 0:
            self.iter_count += 1
            self._update_buffer_num()

        if self.buffer_num > 0 and self.training and input.requires_grad:  # some layers are frozen!
            # cal current batch mu and sigma
            cur_mu = y.mean(dim=1)
            cur_meanx2 = torch.pow(y, 2).mean(dim=1)
            cur_sigma2 = y.var(dim=1)
            # cal dmu/dw dsigma2/dw
            dmudw = torch.autograd.grad(cur_mu, weight, self.ones, retain_graph=True)[0]
            dmeanx2dw = torch.autograd.grad(cur_meanx2, weight, self.ones, retain_graph=True)[0]
            # update cur_mu and cur_sigma2 with pres
            mu_all = torch.stack([cur_mu, ] + [tmp_mu + (self.rho * tmp_d * (weight.data - tmp_w)).sum(1).sum(1).sum(1) for tmp_mu, tmp_d, tmp_w in zip(self.pre_mu, self.pre_dmudw, self.pre_weight)])
            meanx2_all = torch.stack([cur_meanx2, ] + [tmp_meanx2 + (self.rho * tmp_d * (weight.data - tmp_w)).sum(1).sum(1).sum(1) for tmp_meanx2, tmp_d, tmp_w in zip(self.pre_meanx2, self.pre_dmeanx2dw, self.pre_weight)])
            sigma2_all = meanx2_all - torch.pow(mu_all, 2)

            # with considering count
            re_mu_all = mu_all.clone()
            re_meanx2_all = meanx2_all.clone()
            re_mu_all[sigma2_all < 0] = 0
            re_meanx2_all[sigma2_all < 0] = 0
            count = (sigma2_all >= 0).sum(dim=0).float()
            mu = re_mu_all.sum(dim=0) / count
            sigma2 = re_meanx2_all.sum(dim=0) / count - torch.pow(mu, 2)

            self.pre_mu = [cur_mu.detach(), ] + self.pre_mu[:(self.buffer_num - 1)]
            self.pre_meanx2 = [cur_meanx2.detach(), ] + self.pre_meanx2[:(self.buffer_num - 1)]
            self.pre_dmudw = [dmudw.detach(), ] + self.pre_dmudw[:(self.buffer_num - 1)]
            self.pre_dmeanx2dw = [dmeanx2dw.detach(), ] + self.pre_dmeanx2dw[:(self.buffer_num - 1)]

            tmp_weight = torch.zeros_like(weight.data)
            tmp_weight.copy_(weight.data)
            self.pre_weight = [tmp_weight.detach(), ] + self.pre_weight[:(self.buffer_num - 1)]

        else:
            x = y
            mu = x.mean(dim=1)
            cur_mu = mu
            sigma2 = x.var(dim=1)
            cur_sigma2 = sigma2

        if not self.training or self.FROZEN:
            y = y - self.running_mean.view(-1, 1)
            # TODO: outside **0.5?
            if self.out_p:
                y = y / (self.running_var.view(-1, 1) + self.eps)**.5
            else:
                y = y / (self.running_var.view(-1, 1)**.5 + self.eps)
            
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * cur_mu
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * cur_sigma2
            y = y - mu.view(-1, 1)
            # TODO: outside **0.5?
            if self.out_p:
                y = y / (sigma2.view(-1, 1) + self.eps)**.5
            else:
                y = y / (sigma2.view(-1, 1)**.5 + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'buffer={max_buffer_num}, burnin={burnin}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)
