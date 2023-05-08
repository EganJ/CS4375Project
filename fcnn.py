"""
    Implementation of the architecture described by Long, Shelhamer, and Darrell,
    2015 (https://arxiv.org/abs/1411.4038)
"""

import torch
from torch import nn

class BaseFCN(nn.Module):
    def __init__(self, downsample_layers: "list[nn.Module]",
                 prediction_layer: nn.Module,
                 upsample_layers: "list[nn.Module]",
                 final_layer = nn.Softmax2d()):
        super().__init__()

        self.downsample_layers = nn.ModuleList(downsample_layers)
        self.prediction_layer = prediction_layer
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.final_layer = final_layer

    def forward(self, x: torch.Tensor):

        pool_outs = []

        for ds_layer in self.downsample_layers:
            x = ds_layer(x)
            pool_outs.append(x)

        # don't need the last pool output, since the predictions will already
        # be at that level of granularity:
        pool_outs.pop()
        
        x = self.prediction_layer(x)

        for us_layer in self.upsample_layers:
            x = us_layer(x)
            if len(pool_outs) > 0:
                fine_context = pool_outs.pop(-1)
                x = x + fine_context
            
        return self.final_layer(x)


class SimpleFCN(BaseFCN):
    """
        A fully convolutional adaptation of the Simple CNN described in the
        assignment document, under "Candidate CNN models" header.
    """
    def __init__(self, dim_in = 3, dim_out = 4):

        # Downsample to smaller dimensions. We need to keep track of the outputs
        # so they can be fused with upsampling later.
        pool1_dim = 15
        pool2_dim = 15

        pool1_stride = 2
        pool2_stride = 2

        downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim_in, pool1_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.BatchNorm2d(pool1_dim, eps = 1e-05, momentum = 0.1, affine = True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=pool1_stride)),
            nn.Sequential(
                nn.Conv2d(pool1_dim, pool2_dim, kernel_size=(3, 3), stride = (1,1), padding = (1,1)),
                nn.BatchNorm2d(pool2_dim, eps = 1e-05, momentum = 0.1, affine = True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride=pool2_stride))
        ])

        # makes predictions on a coarse level. These would be the linear layers
        # in a simple classification network.
        prediction_layer = nn.Conv2d(pool2_dim, pool2_dim, kernel_size=(7, 7),
                                          stride=(1, 1), padding=(3,3))
        
        upsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=pool2_stride),
                # The paper alludes to this Conv layer but I cannot find an explicit
                # description. This is what makes the most sense to me.
                nn.Conv2d(pool2_dim, pool1_dim, kernel_size=3, padding = 1)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=pool1_stride),
                # The paper alludes to this Conv layer but I cannot find an explicit
                # description. This is what makes the most sense to me.
                nn.Conv2d(pool1_dim, dim_out, kernel_size=3, padding = 1)
            )
        ])

        final_layer = nn.Sigmoid() if dim_out == 1 else nn.Softmax2d()

        super().__init__(downsample_layers, prediction_layer, upsample_layers, final_layer)