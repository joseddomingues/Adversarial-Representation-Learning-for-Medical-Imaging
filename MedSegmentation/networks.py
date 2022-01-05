from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.utils.data


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out


class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """

    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out


class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """

    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out


class R2U_Net(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """

    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)

        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)

        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.RRCNN1(x)

        e2 = self.Maxpool(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool1(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool3(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        # out = self.active(out)

        return out


#
# from __future__ import print_function, division
#
# import torch
# import torch.nn as nn
# import torch.utils.data
#
#
# class U_Net(nn.Module):
#
#     def __init__(self):
#         super(U_Net, self).__init__()
#
#         # Input Tensor Dimensions = 256x256x3
#         # Convolution 1
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
#         nn.init.xavier_uniform_(self.conv1.weight)  # Xaviers Initialisation
#         self.activ_1 = nn.ELU()
#         # Pooling 1
#         self.pool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)
#         # Output Tensor Dimensions = 128x128x16
#
#         # Input Tensor Dimensions = 128x128x16
#         # Convolution 2
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
#         nn.init.xavier_uniform_(self.conv2.weight)
#         self.activ_2 = nn.ELU()
#         # Pooling 2
#         self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)
#         # Output Tensor Dimensions = 64x64x32
#
#         # Input Tensor Dimensions = 64x64x32
#         # Convolution 3
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         nn.init.xavier_uniform_(self.conv3.weight)
#         self.activ_3 = nn.ELU()
#         # Output Tensor Dimensions = 64x64x64
#
#         # 32 channel output of pool2 is concatenated
#
#         # https://www.quora.com/How-do-you-calculate-the-output-dimensions-of-a-deconvolution-network-layer
#         # Input Tensor Dimensions = 64x64x96
#         # De Convolution 1
#         self.deconv1 = nn.ConvTranspose2d(in_channels=96, out_channels=32, kernel_size=3, padding=1)  ##
#         nn.init.xavier_uniform_(self.deconv1.weight)
#         self.activ_4 = nn.ELU()
#         # UnPooling 1
#         self.unpool1 = nn.MaxUnpool2d(kernel_size=2)
#         # Output Tensor Dimensions = 128x128x32
#
#         # 16 channel output of pool1 is concatenated
#
#         # Input Tensor Dimensions = 128x128x48
#         # De Convolution 2
#         self.deconv2 = nn.ConvTranspose2d(in_channels=48, out_channels=16, kernel_size=3, padding=1)
#         nn.init.xavier_uniform_(self.deconv2.weight)
#         self.activ_5 = nn.ELU()
#         # UnPooling 2
#         self.unpool2 = nn.MaxUnpool2d(kernel_size=2)
#         # Output Tensor Dimensions = 256x256x16
#
#         # 3 Channel input is concatenated
#
#         # Input Tensor Dimensions= 256x256x19
#         # DeConvolution 3
#         self.deconv3 = nn.ConvTranspose2d(in_channels=19, out_channels=1, kernel_size=5, padding=2)
#         nn.init.xavier_uniform_(self.deconv3.weight)
#         self.activ_6 = nn.Sigmoid()
#         ##Output Tensor Dimensions = 256x256x1
#
#     def forward(self, x):
#         out_1 = x
#         out = self.conv1(x)
#         out = self.activ_1(out)
#         size1 = out.size()
#         out, indices1 = self.pool1(out)
#         out_2 = out
#         out = self.conv2(out)
#         out = self.activ_2(out)
#         size2 = out.size()
#         out, indices2 = self.pool2(out)
#         out_3 = out
#         out = self.conv3(out)
#         out = self.activ_3(out)
#
#         out = torch.cat((out, out_3), dim=1)
#
#         out = self.deconv1(out)
#         out = self.activ_4(out)
#         out = self.unpool1(out, indices2, size2)
#
#         out = torch.cat((out, out_2), dim=1)
#
#         out = self.deconv2(out)
#         out = self.activ_5(out)
#         out = self.unpool2(out, indices1, size1)
#
#         out = torch.cat((out, out_1), dim=1)
#
#         out = self.deconv3(out)
#         out = self.activ_6(out)
#         out = out
#         return out
#
#
# class up_conv(nn.Module):
#     """
#     Up Convolution Block
#     """
#
#     def __init__(self, in_ch, out_ch):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.up(x)
#         return x
#
#
# class Recurrent_block(nn.Module):
#     """
#     Recurrent Block for R2Unet_CNN
#     """
#
#     def __init__(self, out_ch, t=2):
#         super(Recurrent_block, self).__init__()
#
#         self.t = t
#         self.out_ch = out_ch
#         self.conv = nn.Sequential(
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         for i in range(self.t):
#             if i == 0:
#                 x = self.conv(x)
#             out = self.conv(x + x)
#         return out
#
#
# class RRCNN_block(nn.Module):
#     """
#     Recurrent Residual Convolutional Neural Network Block
#     """
#
#     def __init__(self, in_ch, out_ch, t=2):
#         super(RRCNN_block, self).__init__()
#
#         self.RCNN = nn.Sequential(
#             Recurrent_block(out_ch, t=t),
#             Recurrent_block(out_ch, t=t)
#         )
#         self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         x1 = self.Conv(x)
#         x2 = self.RCNN(x1)
#         out = x1 + x2
#         return out
#
#
# class R2U_Net(nn.Module):
#     """
#     R2U-Unet implementation
#     Paper: https://arxiv.org/abs/1802.06955
#     """
#
#     def __init__(self, img_ch=3, output_ch=1, t=2):
#         super(R2U_Net, self).__init__()
#
#         n1 = 64
#         filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
#
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.Upsample = nn.Upsample(scale_factor=2)
#
#         self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)
#
#         self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)
#
#         self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)
#
#         self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)
#
#         self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)
#
#         self.Up5 = up_conv(filters[4], filters[3])
#         self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)
#
#         self.Up4 = up_conv(filters[3], filters[2])
#         self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)
#
#         self.Up3 = up_conv(filters[2], filters[1])
#         self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)
#
#         self.Up2 = up_conv(filters[1], filters[0])
#         self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)
#
#         self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)
#
#     # self.active = torch.nn.Sigmoid()
#
#     def forward(self, x):
#         e1 = self.RRCNN1(x)
#
#         e2 = self.Maxpool(e1)
#         e2 = self.RRCNN2(e2)
#
#         e3 = self.Maxpool1(e2)
#         e3 = self.RRCNN3(e3)
#
#         e4 = self.Maxpool2(e3)
#         e4 = self.RRCNN4(e4)
#
#         e5 = self.Maxpool3(e4)
#         e5 = self.RRCNN5(e5)
#
#         d5 = self.Up5(e5)
#         d5 = torch.cat((e4, d5), dim=1)
#         d5 = self.Up_RRCNN5(d5)
#
#         d4 = self.Up4(d5)
#         d4 = torch.cat((e3, d4), dim=1)
#         d4 = self.Up_RRCNN4(d4)
#
#         d3 = self.Up3(d4)
#         d3 = torch.cat((e2, d3), dim=1)
#         d3 = self.Up_RRCNN3(d3)
#
#         d2 = self.Up2(d3)
#         d2 = torch.cat((e1, d2), dim=1)
#         d2 = self.Up_RRCNN2(d2)
#
#         out = self.Conv(d2)
#
#         # out = self.active(out)
#
#         return out