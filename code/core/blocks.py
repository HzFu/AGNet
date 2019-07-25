import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS = 1e-4  #1e-4  #1e-5


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=False, BatchNorm=False, is_relu=True, num_groups=32):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        if BatchNorm:
            self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
        if is_bn:
            if out_channels//num_groups==0:
                num_groups=1
            self.gn  =nn.GroupNorm(num_groups, out_channels, eps=BN_EPS)
        self.is_bn = is_bn
        self.is_BatchNorm=BatchNorm
        if is_relu is False: self.relu=None

    def forward(self,x):
        x = self.conv(x)
        if self.is_BatchNorm: x = self.bn(x)
        if self.is_bn: x = self.gn(x)
        if self.relu is not None: x = self.relu(x)
        return x


class StackEncoder (nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3, dilation=1, bn=False, BatchNorm=False, num_groups=32):
        super(StackEncoder, self).__init__()
        padding=(dilation*kernel_size-1)//2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self,x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class StackDecoder (nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, dilation=1, bn=False, BatchNorm=False, num_groups=32):
        super(StackDecoder, self).__init__()
        padding=(dilation*kernel_size-1)//2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels+x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        y = F.upsample(x, size=(H,W),mode='bilinear')
        #y = F.upsample(x, scale_factor=2,mode='bilinear')
        y = torch.cat([y,x_big],1)
        y = self.decode(y)
        return  y


class M_Encoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False, BatchNorm=False, num_groups=32):
        super(M_Encoder, self).__init__()
        padding =(dilation*kernel_size-1)//2
        self.encode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )
        self.pooling = pooling

    def forward(self, x):
        conv = self.encode(x)
        if self.pooling:
            pool = F.max_pool2d(conv, kernel_size=2, stride=2)
            return conv,pool
        else:
            return conv


class M_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False, BatchNorm=False, num_groups=32):
        super(M_Conv, self).__init__()
        padding =(dilation*kernel_size-1)//2
        self.encode = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,kernel_size=kernel_size, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv = self.encode(x)
        return conv


class M_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, deconv = False, bn=False, BatchNorm=False, num_groups=32):
        super(M_Decoder, self).__init__()
        padding =(dilation*kernel_size-1)//2
        if deconv:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=1, padding=1),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                             stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1,is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            )
        else:
            self.deconv = False

        self.decode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        out = F.upsample(x, size=(H,W),mode='bilinear')
        out = torch.cat([x_big,out], dim=1)
        if self.deconv:
            out = self.deconv(out)
        else:
            out = self.decode(out)
        return out


class M_Decoder_my_10(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, deconv = False, bn=False, BatchNorm=False, num_groups=32):
        super(M_Decoder_my_10, self).__init__()
        padding =(dilation*kernel_size-1)//2
        if deconv:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=1, padding=1),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                             stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1,is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            )
        else:
            self.deconv = False

        self.decode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self,x):
        x = self.decode(x)
        return x



