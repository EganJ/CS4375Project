"""
    See https://arxiv.org/pdf/1505.04597.pdf
"""
import torch
from torch import nn

class BaseUNet(nn.Module):
    """
        Difference between this and FCN, as far as I can tell, is the 
        concat vs arithmetic add 
    """
    def __init__(self, layers_down: "list[nn.Module]",
                 pool_down: "list[nn.Module]",
                 prediction_layer: nn.Module,
                 upsamplers: "list[nn.Module]",
                 layers_up,
                 final_layer):
        super().__init__()

        self.layers_down = nn.ModuleList(layers_down)
        # applied AFTER the layer
        self.pool_layers = nn.ModuleList(pool_down)

        self.pred_layer = prediction_layer

        self.layers_up = nn.ModuleList(layers_up)
        # applied BEFORE the layer. The result is concatenated
        self.upsamplers = nn.ModuleList(upsamplers)

        self.final_layer = final_layer


    def forward(self, x: torch.Tensor):
        fine_details = []

        for dl, pool in zip(self.layers_down, self.pool_layers):
            x = dl(x)
            fine_details.append(x)
            x = pool(x)
        
        x = self.pred_layer(x)

        for ul, upsampler in zip(self.layers_up, self.upsamplers):
        
            x = upsampler(x)
            fine = fine_details.pop()
            x = torch.concat([x, fine], dim = 1)
            x = ul(x)

        return self.final_layer(x)



class SimpleUNet(BaseUNet):
    """
    Architecture closely following the paper, i.e. 3x3 conv -> 3x3 conv -> relu
    -> pool layers.
    """

    def __init__(self, channels_down: "list[int]", bottom_channels:int,  channels_up: "list[int]"):
        """
            channel_down[i] is the channels of the inputs to the i'th convolutional
            layer, so channel_down[0] should be the output. 

            bottom_channels is the channels for the convolution at the bottom of the
            u, between downsampling and upsampling.

            channel_up[i] is the OUTPUT of the i'th convolutional layer up, so
            the last entry is the output of the network.
        """

        assert len(channels_down) == len(channels_up)
        n = len(channels_down)

        channels_down = channels_down.copy()
        channels_down.append(bottom_channels)

        channels_up = channels_up.copy()
        channels_up = [bottom_channels] + channels_up
        
        layers_down = []
 
        for i in range(n):
            d_in, d_out = channels_down[i:i+2]
            
            layers_down.append(self.basic_layer(d_in, d_out))

        pool_down = [nn.MaxPool2d(2,2) for _ in layers_down]
        
        pred_layer = self.basic_layer(bottom_channels, bottom_channels)

        layers_up = []
        upsamplers = []

        for i in range(n):

            upsamplers.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(channels_up[i], channels_up[i+1], 3, 1, 1)
            ))

            d_in = channels_up[i+1] + channels_down[-i-1]
            d_out = channels_up[i+1]

            layers_up.append(self.basic_layer(d_in, d_out))

        final_layer = nn.Softmax2d()

        super().__init__(layers_down, pool_down, pred_layer, upsamplers, layers_up, final_layer)
            
    def basic_layer(self, d_in, d_out):
        return nn.Sequential(
            nn.Conv2d(d_in, d_out, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(d_out, d_out, 3, 1, 1),
            nn.ReLU()
        )


def simple_unet_v1(d_in, d_out):
    return SimpleUNet([d_in, 32,32,64], 128, [64, 32, 32, d_out])

def simple_unet_v2(d_in, d_out):
    return SimpleUNet([d_in, 32, 32, 64], 128, [64, 32, 32, d_out])

def simple_unet_paper_version(d_in, d_out):
    return SimpleUNet([d_in, 64,128,256, 512], 1024, [512, 256, 128, 64, d_out])