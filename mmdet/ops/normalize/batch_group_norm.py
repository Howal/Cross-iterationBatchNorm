import torch
import torch.nn as nn
import apex.parallel as ap
from torch.nn.parameter import Parameter
from torch.nn import init


class BatchGroupNorm(nn.Module):
    def __init__(self, *args, num_features, num_groups, affine=True, **kwargs):
        super(BatchGroupNorm, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.affine = affine
        self.inner_bn = nn.BatchNorm1d(num_features=self.num_groups, affine=False, *args, **kwargs)
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
        # [N, G, C//G, H * W]
        input_x = x.view(x.size(0), self.num_groups, self.num_features//self.num_groups, -1)
        # [N, G, C//G * H * W]
        input_x = input_x.view(x.size(0), self.num_groups, -1)
        # [N, G, C//G * H * W]
        output_x = self.inner_bn(input_x)
        # [N, C, H, W]
        output_x = output_x.view_as(x)
        if self.affine:
            output_x = output_x * self.weight.view(1, self.num_features, 1, 1) + \
                       self.bias.view(1, self.num_features, 1, 1)
        return output_x


class SyncBatchGroupNorm(BatchGroupNorm):
    def __init__(self, *args, num_features, num_groups, **kwargs):
        super(SyncBatchGroupNorm, self).__init__(*args, num_features=num_features, num_groups=num_groups, **kwargs)
        self.inner_bn = ap.SyncBatchNorm(num_features=self.num_groups, affine=False, *args, **kwargs)
