import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable, Function

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = Variable(torch.cat([torch.arange(0, 2*N, 2), torch.arange(1, 2*N+1, 2)]), requires_grad=False).type_as(x).long()
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset


class simam_module(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class PSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, parts=4, bias=False):
        super(PSConv2d, self).__init__()
        self.gwconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation, dilation, groups=parts, bias=bias)
        self.gwconv_shift = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 2 * dilation, 2 * dilation, groups=parts, bias=bias)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        def backward_hook(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        self.mask = torch.zeros(self.conv.weight.shape).byte().cuda()
        _in_channels = in_channels // parts
        _out_channels = out_channels // parts
        for i in range(parts):
            self.mask[i * _out_channels: (i + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, : , :] = 1
            self.mask[(i + parts//2)%parts * _out_channels: ((i + parts//2)%parts + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, :, :] = 1
        self.conv.weight.data[self.mask] = 0
        self.conv.weight.register_hook(backward_hook)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x_shift = self.gwconv_shift(torch.cat((x2, x1), dim=1))
        return self.gwconv(x) + self.conv(x) + x_shift


# PSConv-based Group Convolution
class PSGConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, parts=4, bias=False):
        super(PSGConv2d, self).__init__()
        self.gwconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation, dilation, groups=groups * parts, bias=bias)
        self.gwconv_shift = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 2 * dilation, 2 * dilation, groups=groups * parts, bias=bias)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

        def backward_hook(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        self.mask = torch.zeros(self.conv.weight.shape).byte().cuda()
        _in_channels = in_channels // (groups * parts)
        _out_channels = out_channels // (groups * parts)
        for i in range(parts):
            for j in range(groups):
                self.mask[(i + j * groups) * _out_channels: (i + j * groups + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, : , :] = 1
                self.mask[((i + parts // 2) % parts + j * groups) * _out_channels: ((i + parts // 2) % parts + j * groups + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, :, :] = 1
        self.conv.weight.data[self.mask] = 0
        self.conv.weight.register_hook(backward_hook)
        self.groups = groups

    def forward(self, x):
        x_split = (z.chunk(2, dim=1) for z in x.chunk(self.groups, dim=1))
        x_merge = torch.cat(tuple(torch.cat((x2, x1), dim=1) for (x1, x2) in x_split), dim=1)
        x_shift = self.gwconv_shift(x_merge)
        return self.gwconv(x) + self.conv(x) + x_shift
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size -1 ) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
class RefConv(nn.Module):
    """
    Implementation of RefConv.
    --in_channels: number of input channels in the basis kernel
    --out_channels: number of output channels in the basis kernel
    --kernel_size: size of the basis kernel
    --stride: stride of the original convolution
    --padding: padding added to all four sides of the basis kernel
    --groups: groups of the original convolution
    --map_k: size of the learnable kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1, map_k=3):
        super(RefConv, self).__init__()
        assert map_k <= kernel_size
        self.origin_kernel_shape = (out_channels, in_channels // groups,kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels, out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2, groups=G, bias=False)
        #nn.init.zeros_(self.convmap.weight)
        #zero initialization the trainable weights
        self.bias = None
        #nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding
    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding,dilation=1, groups=self.groups, bias=self.bias)

# （2）空间注意力机制
class spatial_attention(nn.Module):
    # 卷积核大小为7*7
    def __init__(self, kernel_size=3):
        super().__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2

        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)

        # 空间权重归一化
        x = self.sigmoid(x)

        # 输入特征图和空间权重相乘
        outputs = inputs * x

        return outputs


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        # 这一行千万不要忘记
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)


    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out