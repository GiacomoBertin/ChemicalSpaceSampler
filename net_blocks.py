from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from typing import List


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


# fc layers
def fc(batch_norm, nc_inp, nc_out, activation=None):
    if batch_norm:
        return nn.Sequential(
            nn.Linear(nc_inp, nc_out, bias=True),
            nn.BatchNorm1d(nc_out),
            nn.LeakyReLU(0.2, inplace=True) if activation is None else activation
        )
    else:
        return nn.Sequential(
            nn.Linear(nc_inp, nc_out),
            nn.LeakyReLU(0.1, inplace=True) if activation is None else activation
        )


def fc_stack(nc_inp, nc_out, nlayers, use_bn=True):
    modules = []
    for l in range(nlayers):
        modules.append(fc(use_bn, nc_inp, nc_out))
        nc_inp = nc_out
    encoder = nn.Sequential(*modules)
    net_init(encoder)
    return encoder


# 2D convolution layers
def conv2d(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


# 2D convolution layers
def conv1d(batch_norm, in_planes, out_planes, kernel_size=3, stride=1, activation=None):
    if batch_norm:
        return nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.BatchNorm1d(out_planes),
            nn.LeakyReLU(0.2, inplace=True) if activation is None else activation
        )
    else:
        return nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.LeakyReLU(0.2, inplace=True) if activation is None else activation
        )


def deconv2d(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.2, inplace=True)
    )


def deconv1d(in_planes, out_planes, activation=None):
    return nn.Sequential(
        nn.ConvTranspose1d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.2, inplace=True) if activation is None else activation
    )


def upconv2d(in_planes, out_planes, mode='bilinear'):
    if mode == 'nearest':
        print('Using NN upsample!!')
    upconv = nn.Sequential(
        nn.Upsample(scale_factor=2, mode=mode),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return upconv


def upconv1d(in_planes, out_planes, mode='bilinear'):
    if mode == 'nearest':
        print('Using NN upsample!!')
    upconv = nn.Sequential(
        nn.Upsample(scale_factor=2, mode=mode),
        nn.ReflectionPad2d(1),
        nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return upconv


def downconv1d(batch_norm, in_planes, out_planes, pooling='AvgPool1d'):
    downconv = nn.Sequential(
        getattr(nn, pooling)(2),
        nn.ReflectionPad2d(1),
        nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm1d(out_planes) if batch_norm else nn.Identity(),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return downconv


def decoder2d(nlayers, nz_shape, nc_input, use_bn=True, nc_final=1, nc_min=8, nc_step=1, init_fc=True, use_deconv=False,
              upconv_mode='bilinear'):
    """ Simple 3D encoder with nlayers.

    Args:
        nlayers: number of decoder layers
        nz_shape: number of bottleneck
        nc_input: number of channels to start upconvolution from
        use_bn: whether to use batch_norm
        nc_final: number of output channels
        nc_min: number of min channels
        nc_step: double number of channels every nc_step layers
        init_fc: initial features are not spatial, use an fc & unsqueezing to make them 3D
        upconv_mode:
        use_deconv:
    """
    modules = []
    if init_fc:
        modules.append(fc(use_bn, nz_shape, nc_input))
        for d in range(3):
            modules.append(Unsqueeze(2))
    nc_output = nc_input
    for nl in range(nlayers):
        if (nl % nc_step == 0) and (nc_output // 2 >= nc_min):
            nc_output = nc_output // 2
        if use_deconv:
            print('Using deconv decoder!')
            modules.append(deconv2d(nc_input, nc_output))
            nc_input = nc_output
            modules.append(conv2d(use_bn, nc_input, nc_output))
        else:
            modules.append(upconv2d(nc_input, nc_output, mode=upconv_mode))
            nc_input = nc_output
            modules.append(conv2d(use_bn, nc_input, nc_output))

    modules.append(nn.Conv2d(nc_output, nc_final, kernel_size=3, stride=1, padding=1, bias=True))
    decoder = nn.Sequential(*modules)
    net_init(decoder)
    return decoder


# 3D convolution layers
def conv3d(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.BatchNorm3d(out_planes),
            nn.LeakyReLU(0.2, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


def deconv3d(batch_norm, in_planes, out_planes):
    if batch_norm:
        return nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm3d(out_planes),
            nn.LeakyReLU(0.2, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


# 3D Network Modules
def encoder3d(nlayers, use_bn=True, nc_input=1, nc_max=128, nc_l1=8, nc_step=1, nz_shape=20):
    ''' Simple 3D encoder with nlayers.

    Args:
        nlayers: number of encoder layers
        use_bn: whether to use batch_norm
        nc_input: number of input channels
        nc_max: number of max channels
        nc_l1: number of channels in layer 1
        nc_step: double number of channels every nc_step layers
        nz_shape: size of bottleneck layer
    '''
    modules = []
    nc_output = nc_l1
    for nl in range(nlayers):
        if (nl >= 1) and (nl % nc_step == 0) and (nc_output <= nc_max * 2):
            nc_output *= 2

        modules.append(conv3d(use_bn, nc_input, nc_output, stride=1))
        nc_input = nc_output
        modules.append(conv3d(use_bn, nc_input, nc_output, stride=1))
        modules.append(torch.nn.MaxPool3d(kernel_size=2, stride=2))

    modules.append(Flatten())
    modules.append(fc_stack(nc_output, nz_shape, 2, use_bn=True))
    encoder = nn.Sequential(*modules)
    net_init(encoder)
    return encoder, nc_output


def decoder3d(nlayers, nz_shape, nc_input, use_bn=True, nc_final=1, nc_min=8, nc_step=1, init_fc=True):
    ''' Simple 3D encoder with nlayers.

    Args:
        nlayers: number of decoder layers
        nz_shape: number of bottleneck
        nc_input: number of channels to start upconvolution from
        use_bn: whether to use batch_norm
        nc_final: number of output channels
        nc_min: number of min channels
        nc_step: double number of channels every nc_step layers
        init_fc: initial features are not spatial, use an fc & unsqueezing to make them 3D
    '''
    modules = []
    if init_fc:
        modules.append(fc(use_bn, nz_shape, nc_input))
        for d in range(3):
            modules.append(Unsqueeze(2))
    nc_output = nc_input
    for nl in range(nlayers):
        if (nl % nc_step == 0) and (nc_output // 2 >= nc_min):
            nc_output = nc_output // 2

        modules.append(deconv3d(use_bn, nc_input, nc_output))
        nc_input = nc_output
        modules.append(conv3d(use_bn, nc_input, nc_output))

    modules.append(nn.Conv3d(nc_output, nc_final, kernel_size=3, stride=1, padding=1, bias=True))
    decoder = nn.Sequential(*modules)
    net_init(decoder)
    return decoder


def normal_init(tensor, mean=0., std=0.02):
    tensor.normal_(mean, std)


def xavier_normal_init(tensor, non_linearity='leaky_relu'):
    gain = nn.init.calculate_gain(non_linearity, 0.2)
    nn.init.xavier_normal_(tensor, gain=gain)


def xavier_uniform_init(tensor, non_linearity='leaky_relu'):
    gain = nn.init.calculate_gain(non_linearity, 0.2)
    nn.init.xavier_uniform_(tensor, gain=gain)


def kaiming_normal_init(tensor, non_linearity='leaky_relu'):
    nn.init.kaiming_normal_(tensor, a=0.2, nonlinearity=non_linearity)


def kaiming_uniform_init(tensor, non_linearity='leaky_relu'):
    nn.init.kaiming_uniform_(tensor, a=0.2, nonlinearity=non_linearity)


def net_init(net, zero_bias=True, net_init_func=normal_init):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            net_init_func(m.weight.data)
            if m.bias is not None and zero_bias:
                m.bias.data.zero_()

        if isinstance(m, nn.Conv2d):  # or isinstance(m, nn.ConvTranspose2d):
            net_init_func(m.weight.data)
            if m.bias is not None and zero_bias:
                m.bias.data.zero_()

        if isinstance(m, nn.ConvTranspose2d):
            # Initialize Deconv with bilinear weights.
            base_weights = bilinear_init(m.weight.data.size(-1))
            base_weights = base_weights.unsqueeze(0).unsqueeze(0)
            m.weight.data = base_weights.repeat(m.weight.data.size(0), m.weight.data.size(1), 1, 1)
            if m.bias is not None and zero_bias:
                m.bias.data.zero_()

        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            net_init_func(m.weight.data)
            if m.bias is not None and zero_bias:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight.data, 1)
            m.bias.data.zero_()

    return net


def bilinear_init(kernel_size=4):
    # Following Caffe's BilinearUpsamplingFiller
    # https://github.com/BVLC/caffe/pull/2213/files
    import numpy as np
    width = kernel_size
    height = kernel_size
    f = int(np.ceil(width / 2.))
    cc = (2 * f - 1 - f % 2) / (2. * f)
    weights = torch.zeros((height, width))
    for y in range(height):
        for x in range(width):
            weights[y, x] = (1 - np.abs(x / f - cc)) * (1 - np.abs(y / f - cc))

    return weights


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, label_nc, input_ch, hidden_ch=128):
        super(ConditionalBatchNorm2d, self).__init__()

        self.param_free_norm = nn.BatchNorm2d(input_ch)
        self.mlp_shared = nn.Sequential(nn.Linear(label_nc, input_ch), nn.ReLU())
        self.mlp_gamma = nn.Linear(hidden_ch, input_ch)
        self.mlp_beta = nn.Linear(hidden_ch, input_ch)

    def forward(self, x, c):
        normalized = self.param_free_norm(x)
        actv = self.mlp_shared(c)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma[..., None, None]) + beta[..., None, None]
        return out


class UpBlock(nn.Module):
    def __init__(self, input_ch, output_ch, label_ch=None):
        super(UpBlock, self).__init__()
        self.use_cbn = label_ch is not None
        self.conv1 = nn.Conv2d(input_ch, output_ch,
                               kernel_size=3,
                               stride=1,
                               padding=1
                               )
        self.upsample = nn.Upsample(scale_factor=2)
        self.bn = nn.BatchNorm2d(output_ch) if label_ch is None else ConditionalBatchNorm2d(label_ch, output_ch)
        self.act = nn.ReLU()

    def forward(self, x, c=None):
        h = self.upsample(x)
        h = self.conv1(h)
        h = self.bn(h, c) if self.use_cbn else self.bn(h)
        h = self.act(h)
        return h

