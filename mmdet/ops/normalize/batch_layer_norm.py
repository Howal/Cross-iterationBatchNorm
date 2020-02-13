import torch
import torch.nn as nn
import apex.parallel as ap
from torch.nn.parameter import Parameter
from torch.nn import init


class BatchLayerNorm(nn.Module):
    def __init__(self, *args, num_features, affine=True, **kwargs):
        super(BatchLayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.inner_bn = nn.BatchNorm1d(num_features=1, affine=False, *args, **kwargs)
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.uniform_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x):
        # [N, 1, C * H * W]
        input_x = x.view(x.size(0), 1, -1)
        # [N, 1, C * H * W]
        output_x = self.inner_bn(input_x)
        # [N, C, H, W]
        output_x = output_x.view_as(x)
        if self.affine:
            output_x = output_x * self.weight.view(1, self.num_features, 1, 1) + \
                       self.bias.view(1, self.num_features, 1, 1)
        return output_x


class SyncBatchLayerNorm(BatchLayerNorm):
    def __init__(self, *args, num_features, **kwargs):
        super(SyncBatchLayerNorm, self).__init__(*args, num_features=num_features, **kwargs)
        self.inner_bn = ap.SyncBatchNorm(num_features=1, affine=False, *args, **kwargs)
