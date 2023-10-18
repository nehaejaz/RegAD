# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import numpy as np

N_PARAMS = {'affine': 6,
            'translation': 2,
            'rotation': 1,
            'scale': 2,
            'shear': 2,
            'rotation_scale': 3,
            'translation_scale': 4,
            'rotation_translation': 3,
            'rotation_translation_scale': 5}

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
    
class STNModule(nn.Module):
    def __init__(self, in_num, block_index, args):
        super(STNModule, self).__init__()

        self.feat_size = block_index
        self.stn_mode = "rotation_scale"
        self.stn_n_params = N_PARAMS[self.stn_mode]
        self.in_num = in_num
        self.conv = nn.Sequential(
            conv3x3(in_planes=in_num, out_planes=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv3x3(in_planes=64, out_planes=16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=16 * self.feat_size * self.feat_size, out_features=1024),
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=self.stn_n_params),
        )
        
        self.fc[2].weight.data.fill_(0)
        self.fc[2].weight.data.zero_()

        if self.stn_mode == 'affine':
            self.fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif self.stn_mode in ['translation', 'shear']:
            self.fc[2].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        elif self.stn_mode == 'scale':
            self.fc[2].bias.data.copy_(torch.tensor([1, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation':
            self.fc[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))
        elif self.stn_mode == 'rotation_scale':
            self.fc[2].bias.data.copy_(torch.tensor([0, 1, 1], dtype=torch.float))
        elif self.stn_mode == 'translation_scale':
            self.fc[2].bias.data.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation':
            self.fc[2].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation_scale':
            self.fc[2].bias.data.copy_(torch.tensor([0, 0, 0, 1, 1], dtype=torch.float))

    def forward(self, x):
        # print("feat_size",self.feat_size)
        # print("in_num",self.in_num)
        mode = self.stn_mode
        batch_size = x.size(0)
        conv_x = self.conv(x)
        # print("conv_x shape",conv_x.shape)
        theta = self.fc(conv_x.view(batch_size, -1))
        # print("theta", theta.shape)

        if mode == 'affine':
            theta1 = theta.view(batch_size, 2, 3)
        else:
            theta1 = Variable(torch.zeros([batch_size, 2, 3], dtype=torch.float32, device=x.get_device()),
                              requires_grad=True)
            theta1 = theta1 + 0
            theta1[:, 0, 0] = 1.0
            theta1[:, 1, 1] = 1.0
            if mode == 'translation':
                theta1[:, 0, 2] = theta[:, 0]
                theta1[:, 1, 2] = theta[:, 1]
            elif mode == 'rotation':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle)
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle)
            elif mode == 'scale':
                theta1[:, 0, 0] = theta[:, 0]
                theta1[:, 1, 1] = theta[:, 1]
            elif mode == 'shear':
                theta1[:, 0, 1] = theta[:, 0]
                theta1[:, 1, 0] = theta[:, 1]
            elif mode == 'rotation_scale':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle) * theta[:, 1]
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle) * theta[:, 2]
            elif mode == 'translation_scale':
                theta1[:, 0, 2] = theta[:, 0]
                theta1[:, 1, 2] = theta[:, 1]
                theta1[:, 0, 0] = theta[:, 2]
                theta1[:, 1, 1] = theta[:, 3]
            elif mode == 'rotation_translation':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle)
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle)
                theta1[:, 0, 2] = theta[:, 1]
                theta1[:, 1, 2] = theta[:, 2]
            elif mode == 'rotation_translation_scale':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle) * theta[:, 3]
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle) * theta[:, 4]
                theta1[:, 0, 2] = theta[:, 1]
                theta1[:, 1, 2] = theta[:, 2]

        grid = F.affine_grid(theta1, torch.Size(x.shape))
        img_transform = F.grid_sample(x, grid, padding_mode="reflection")

        return img_transform, theta1

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, args, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], stn=True, drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1., 
                 ):
        super().__init__()
        self.args = "rotation_scale"
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.output_layers = []
        self.stn = stn
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stn_modules = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks

        # self.stn1 = STNModule(96, 1, args)
        # self.stn2 = STNModule(192, 2, args)
        # self.stn3 = STNModule(384, 3, args)
        # self.stn3 = STNModule(768, 4, args)


        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        block_num = [14,7,4,2]
        for i in range(4):
            #Adding sequential layer
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            #Appending that into stages
            self.stages.append(stage)
            
            if self.stn is True:
                #Adding stn layer
                stn = STNModule(dims[i], block_num[i] , "rotation_scale")

                #Appending that into stages
                self.stages.append(stn)

            cur += depths[i]
     
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if(m.bias is not None): 
                nn.init.constant_(m.bias, 0)
                
    def _fixstn(self, x, theta):
        grid = F.affine_grid(theta, torch.Size(x.shape))
        img_transform = F.grid_sample(x, grid, padding_mode="reflection")
        return img_transform

    def forward_features(self, x):
        self.output_layers = []
        ds_lay_in = 0
        if self.stn is True:

            for i in range(0,8,2):
                x = self.downsample_layers[ds_lay_in](x)
                x = self.stages[i](x)
                # print("before",x.shape)
                x, theta = self.stages[i+1](x)
                tmp = np.tile(np.array([0, 0, 1]), (x.shape[0], 1, 1)).astype(np.float32)
                fixtheata = torch.from_numpy(np.linalg.inv(np.concatenate((theta.detach().cpu().numpy(), tmp), axis=1))[:,:-1,:]).cuda(3)
                x = self._fixstn(x.detach(), fixtheata)
                # print("after",x.shape)
                self.output_layers.append(x)
                ds_lay_in += 1
        else:
            for i in range(4):
                x = self.downsample_layers[ds_lay_in](x)
                x = self.stages[i](x)
                # print("before",x.shape)
                self.output_layers.append(x)
                ds_lay_in += 1
        # return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(args, pretrained=True,in_22k=False, **kwargs):
    model = ConvNeXt(args, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    # print(model)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        # model.load_state_dict(checkpoint["model"])
        model.load_state_dict(model_zoo.load_url(model_urls['convnext_tiny_22k']), strict=False)
        # Get the state dictionary of the model
        state_dict = model.state_dict()

        # Print the keys
        # print(state_dict.keys())
        # print(model)
    return model

@register_model
def convnext_small(args,pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(args,depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

# model = convnext_tiny("rotation_scale")
# print(model)