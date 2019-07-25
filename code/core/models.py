import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import M_Encoder
from blocks import M_Conv

from blocks import M_Decoder_my_10
from guided_filter_pytorch.guided_filter_attention import FastGuidedFilter_attention


class AG_Net(nn.Module):
    def __init__(self, n_classes, bn=True, BatchNorm=False):
        super(AG_Net, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter_attention(r=2, eps=1e-2)

        # attention blocks
        self.attentionblock5 = GridAttentionBlock(in_channels=512)
        self.attentionblock6 = GridAttentionBlock(in_channels=256)
        self.attentionblock7 = GridAttentionBlock(in_channels=128)
        self.attentionblock8 = GridAttentionBlock(in_channels=64)


    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)

        FG = torch.cat([self.conv4(x_4), conv4], dim=1)
        N, C, H, W= FG.size()
        FG_small = F.upsample(FG, size=(H/2, W/2), mode='bilinear')
        out = self.gf(FG_small, out, FG,self.attentionblock5(FG_small,out))
        up5 = self.up5(out)

        FG = torch.cat([self.conv3(x_3), conv3], dim=1)
        N, C, H, W = FG.size()
        FG_small = F.upsample(FG, size=(H/2, W/2), mode='bilinear')
        out = self.gf(FG_small, up5, FG,self.attentionblock6(FG_small,up5))
        up6 = self.up6(out)

        FG = torch.cat([self.conv2(x_2), conv2], dim=1)
        N, C, H, W = FG.size()
        FG_small = F.upsample(FG, size=(H/2, W/2), mode='bilinear')
        out = self.gf(FG_small, up6, FG,self.attentionblock7(FG_small,up6))
        up7 = self.up7(out)

        FG = torch.cat([conv1, conv1], dim=1)
        N, C, H, W = FG.size()
        FG_small = F.upsample(FG, size=(H/2, W/2), mode='bilinear')
        out = self.gf(FG_small, up7, FG,self.attentionblock8(FG_small,up7))
        up8 = self.up8(out)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class GridAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(GridAttentionBlock, self).__init__()

        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1)

        self.phi = nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear')
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.sigmoid(self.psi(f))

        return sigm_psi_f

