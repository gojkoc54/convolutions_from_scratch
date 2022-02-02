import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
from torch.nn.modules.utils import _pair

import numpy as np
import copy 


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.stride = stride 
        self.kernel_size = kernel_size
        self.padding = padding

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=kernel_size, stride=stride, 
                padding=padding, bias=False
                ), 
            nn.BatchNorm2d(out_channels, affine=False), 
            nn.ReLU()
            )

    def forward(self, x):
        out = self.layers(x)
        return out 

    def _calculate_out_size(self, in_size):
        """
        in_size = (in_width, in_height)
        """
        out_width = in_size[0] + 2 * self.padding - self.kernel_size 
        out_width = int(out_width / self.stride) + 1
        out_height = in_size[1] + 2 * self.padding - self.kernel_size
        out_height = int(out_height / self.stride) + 1
        return (out_width, out_height)


class FCBlock(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super(FCBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=bias), 
            nn.BatchNorm1d(out_dim, affine=False), 
            nn.ReLU()
            )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.layers(out)
        return out



class SConv(nn.Module):
    def _calculate_out_size(self):
        return self.conv_layer._calculate_out_size(self.img_size)

    def __init__(self, in_channels, base_channels, img_size, **kwargs):
        super(SConv, self).__init__()

        self.img_size = img_size
        self.kernel_size = kwargs['kernel_size'] if 'kernel_size' in kwargs else 9
        self.padding = kwargs['padding'] if 'padding' in kwargs else 0
        self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 10

        self.conv_layer = ConvBlock(
            in_channels, base_channels, 
            kernel_size=self.kernel_size, stride=2, padding=self.padding
            )
        
        # Calculate the output size of the convolutional blocks, i.e. the input
        # size to the fully-connected layer
        conv_out_width, conv_out_height = self._calculate_out_size()
        fc_input_dim = base_channels * conv_out_width * conv_out_height
        
        # First FC is FCBlock (with BN and ReLU) and second just a Linear layer
        self.fc_layers = nn.Sequential(
            FCBlock(fc_input_dim, 24 * base_channels), 
            nn.Linear(24 * base_channels, self.num_classes, bias=False)
            )

    def forward(self, x):
        out = self.conv_layer(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)

        return out 



class DConv(nn.Module):
    def _calculate_out_size(self):
        out_dim = self.img_size
        for i in range(len(self.conv_layers)):
            out_dim = self.conv_layers[i]._calculate_out_size(out_dim)

        return out_dim 

    def __init__(self, in_channels, base_channels, img_size, **kwargs):
        super(DConv, self).__init__()
        
        self.img_size = img_size
        self.kernel_size = kwargs['kernel_size'] if 'kernel_size' in kwargs else 3
        self.padding = kwargs['padding'] if 'padding' in kwargs else 0
        self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 10

        self.conv_layers = nn.ModuleList()

        # First layer - (alpha, 1)
        # in_channels -> base_channels
        self.conv_layers.append(
            ConvBlock(
                in_channels, base_channels, 
                kernel_size=self.kernel_size, stride=1, padding=self.padding
                )
            )

        num_block_pairs = 3
        scales = [2**i for i in range(num_block_pairs + 2)]
        channels_per_layer = [scale * base_channels for scale in scales]
        for i in range(len(channels_per_layer) - 2):
            # base_channels, 2 * base_channels, 4 * base_channels
            curr_in_channels = channels_per_layer[i]
            # 2 * base_channels, 4 * base_channels, 8 * base_channels
            curr_out_channels = channels_per_layer[i + 1]
            
            # Even block - (scale * alpha, 2)
            self.conv_layers.append(
                ConvBlock(
                    curr_in_channels, curr_out_channels, 
                    kernel_size=self.kernel_size, stride=2, padding=self.padding
                    )
                )
            
            # Odd block - (scale * alpha, 1)
            self.conv_layers.append(
                ConvBlock(
                    curr_out_channels, curr_out_channels, 
                    kernel_size=self.kernel_size, stride=1, padding=self.padding
                    )
                )

        self.conv_layers.append(
            ConvBlock(
                scales[-2] * base_channels, scales[-1] * base_channels, 
                kernel_size=self.kernel_size, stride=2, padding=self.padding
                )
            )

        conv_out_width, conv_out_height = self._calculate_out_size()
        fc_input_dim = 16 * base_channels * conv_out_width * conv_out_height
        
        self.fc_layers = nn.Sequential(
            FCBlock(fc_input_dim, 64 * base_channels),
            nn.Linear(64 * base_channels, self.num_classes, bias=False)
            )


    def forward(self, x):
        out = copy.deepcopy(x)
        for i in range(len(self.conv_layers)):
            out = self.conv_layers[i](out)

        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)

        return out 



class DFC(nn.Module):
    def __init__(self, in_channels, base_channels, img_size, num_classes=10):
        super(DFC, self).__init__()

        self.img_size = img_size
        self.num_classes = num_classes

        self.fc_layers = nn.ModuleList()

        img_dim = img_size[0] * img_size[1]
        self.fc_layers.append(
            FCBlock(in_channels * img_dim, base_channels * img_dim)
            )

        num_block_pairs = 3
        scales = [2**(-i) for i in range(num_block_pairs + 2)]
        dim_per_layer = [int(scale * base_channels * img_dim) for scale in scales]
        for i in range(len(dim_per_layer) - 2):
            curr_in_dim, curr_out_dim = dim_per_layer[i], dim_per_layer[i + 1]

            # Even block
            self.fc_layers.append(FCBlock(curr_in_dim, curr_out_dim))
            
            # Odd block 
            self.fc_layers.append(FCBlock(curr_out_dim, curr_out_dim))

        self.fc_layers.append(FCBlock(dim_per_layer[-2], dim_per_layer[-1]))
            
        # Last two layers
        self.fc_layers.append(
            nn.Sequential(
                FCBlock(dim_per_layer[-1], 64 * base_channels),
                nn.Linear(64 * base_channels, num_classes, bias=False)
                )
            )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        for i in range(len(self.fc_layers)):
            out = self.fc_layers[i](out)

        return out
                    
        

class SFC(nn.Module):
    def __init__(self, in_channels, base_channels, img_size, num_classes=10):
        super(SFC, self).__init__()

        self.img_size = img_size
        self.num_classes = num_classes

        self.fc_layers = nn.ModuleList()

        img_dim = img_size[0] * img_size[1]
        self.fc_layers.append(
            FCBlock(in_channels * img_dim, int(base_channels / 4 * img_dim))
            )
        
        # Last two layers
        self.fc_layers.append(
            nn.Sequential(
                FCBlock(int(base_channels / 4 * img_dim), 24 * base_channels),
                nn.Linear(24 * base_channels, num_classes, bias=False)
                )
            )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        for i in range(len(self.fc_layers)):
            out = self.fc_layers[i](out)

        return out



class LocallyConnected2d(nn.Module):
    def _calculate_out_size(self, in_size):
        """
        in_size = (in_width, in_height)
        """
        out_width = in_size[0] + 2 * self.padding[0] - self.kernel_size[0]
        out_width = int(out_width / self.stride[0]) + 1
        out_height = in_size[1] + 2 * self.padding[1] - self.kernel_size[1]
        out_height = int(out_height / self.stride[1]) + 1
        return (out_width, out_height)

    def __init__(self, in_channels, out_channels, in_size, **kwargs):
        super(LocallyConnected2d, self).__init__()

        self.in_size = in_size
        self.kernel_size = _pair(
            kwargs['kernel_size'] if 'kernel_size' in kwargs else 3
            )
        self.padding = _pair(kwargs['padding'] if 'padding' in kwargs else 0)
        self.stride = _pair(kwargs['stride'] if 'stride' in kwargs else 1)
        self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 10
        self.register_parameter('bias', None)

        output_size = self._calculate_out_size(self.in_size)

        self.weight = nn.Parameter(
            torch.randn(
                1, out_channels, in_channels, 
                output_size[0], output_size[1], 
                self.kernel_size[0] * self.kernel_size[1]
                )
            )

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        padding_2dim = self.padding + self.padding

        x = nn.functional.pad(x, padding_2dim)
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)

        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])

        return out



class LocalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, **kwargs):
        super(LocalBlock, self).__init__()

        self.img_size = img_size
        self.kernel_size = kwargs['kernel_size'] if 'kernel_size' in kwargs else 3
        self.padding = kwargs['padding'] if 'padding' in kwargs else 0
        self.stride = kwargs['stride'] if 'stride' in kwargs else 1

        self.layers = nn.Sequential(
            LocallyConnected2d(
                in_channels, out_channels, self.img_size,
                kernel_size=self.kernel_size, stride=self.stride, 
                padding=self.padding
                ), 
            nn.BatchNorm2d(out_channels, affine=False), 
            nn.ReLU()
            )

    def forward(self, x):
        out = self.layers(x)
        return out 

    def _calculate_out_size(self, in_size):
        return self.layers[0]._calculate_out_size(in_size)



class SLocal(nn.Module):
    def _calculate_out_size(self):
        return self.local_layer._calculate_out_size(self.img_size)

    def __init__(self, in_channels, base_channels, img_size, **kwargs):
        super(SLocal, self).__init__()

        self.img_size = img_size
        self.kernel_size = kwargs['kernel_size'] if 'kernel_size' in kwargs else 9
        self.padding = kwargs['padding'] if 'padding' in kwargs else 0
        self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 10

        self.local_layer = LocalBlock(
            in_channels, base_channels, self.img_size,
            kernel_size=self.kernel_size, 
            stride=2, padding=self.padding
            )
        
        local_out_width, local_out_height = self._calculate_out_size()
        fc_input_dim = base_channels * local_out_width * local_out_height
        
        # First FC is FCBlock (with BN and ReLU) and second just a Linear layer
        self.fc_layers = nn.Sequential(
            FCBlock(fc_input_dim, 24 * base_channels), 
            nn.Linear(24 * base_channels, self.num_classes, bias=False)
            )

    def forward(self, x):
        out = self.local_layer(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)

        return out 



class DLocal(nn.Module):
    def _calculate_out_size(self):
        out_dim = self.img_size
        for i in range(len(self.local_layers)):
            out_dim = self.local_layers[i]._calculate_out_size(out_dim)

        return out_dim 

    def __init__(self, in_channels, base_channels, img_size, **kwargs):
        super(DLocal, self).__init__()

        self.img_size = img_size
        self.kernel_size = kwargs['kernel_size'] if 'kernel_size' in kwargs else 3
        self.padding = kwargs['padding'] if 'padding' in kwargs else 0
        self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 10

        self.local_layers = nn.ModuleList()
        curr_img_size = copy.copy(self.img_size)

        # First layer - (alpha, 1)
        # in_channels -> base_channels
        self.local_layers.append(
            LocalBlock(
                in_channels, base_channels, curr_img_size,
                kernel_size=self.kernel_size, 
                stride=1, padding=self.padding
                )
            )
        curr_img_size = self.local_layers[-1]._calculate_out_size(curr_img_size)

        num_block_pairs = 3
        scales = [2**i for i in range(num_block_pairs + 2)]
        channels_per_layer = [scale * base_channels for scale in scales]
        for i in range(len(channels_per_layer) - 2):
            # base_channels, 2 * base_channels, 4 * base_channels
            curr_in_channels = channels_per_layer[i]
            # 2 * base_channels, 4 * base_channels, 8 * base_channels
            curr_out_channels = channels_per_layer[i + 1]
            
            # Even block - (alpha, alpha/2, alpha/4)
            self.local_layers.append(
                LocalBlock(
                    curr_in_channels, curr_out_channels, curr_img_size,
                    kernel_size=self.kernel_size, 
                    stride=2, padding=self.padding
                    )
                )
            curr_img_size = self.local_layers[-1]._calculate_out_size(curr_img_size)
            
            # Odd block - (alpha/2, alpha/4, alpha/8)
            self.local_layers.append(
                LocalBlock(
                    curr_out_channels, curr_out_channels, curr_img_size,
                    kernel_size=self.kernel_size, 
                    stride=1, padding=self.padding
                    )
                )
            curr_img_size = self.local_layers[-1]._calculate_out_size(curr_img_size)


        self.local_layers.append(
            LocalBlock(
                channels_per_layer[-2], channels_per_layer[-1], curr_img_size, 
                kernel_size=self.kernel_size, 
                stride=2, padding=self.padding
                )
            )
        curr_img_size = self.local_layers[-1]._calculate_out_size(curr_img_size)

        local_out_width, local_out_height = curr_img_size
        fc_input_dim = 16 * base_channels * local_out_width * local_out_height
        
        self.fc_layers = nn.Sequential(
            FCBlock(fc_input_dim, 64 * base_channels),
            nn.Linear(64 * base_channels, self.num_classes, bias=False)
            )
        
    def forward(self, x):
        out = copy.deepcopy(x)
        for i in range(len(self.local_layers)):
            out = self.local_layers[i](out)

        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)

        return out 
    


####################################
##### !!! WORK IN PROGRESS !!! #####
####################################
class ThreeFC(nn.Module):
    def __init__(self, in_channels, base_channels, img_size, num_classes=10):
        super(ThreeFC, self).__init__()

        self.img_size = img_size
        self.num_classes = num_classes

        # Calculate hidden layer dimension
        # ==>> 2 layers of dim = (SFC_hidden_dim) / 2
        hidden_dim = 3 * img_size[0]**4 / 4 * base_channels 
        hidden_dim += 6 * img_size[0]**2 * base_channels**2
        hidden_dim = int(hidden_dim / 2)

        self.fc_layers = nn.ModuleList()

        img_dim = img_size[0] * img_size[1]
        self.fc_layers.append(
            FCBlock(in_channels * img_dim, hidden_dim)
            )
        
        # Last two layers
        self.fc_layers.append(
            nn.Sequential(
                FCBlock(hidden_dim, 24 * base_channels),
                nn.Linear(24 * base_channels, num_classes, bias=False)
                )
            )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        for i in range(len(self.fc_layers)):
            out = self.fc_layers[i](out)

        return out
