import torch
import torch.nn as nn
import torch.nn.functional as F


class SideWindowConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=False, padding_mode='zeros'):
        super(SideWindowConv, self).__init__()

        assert kernel_size>=3, 'kernel size must great than 3'
        self.radius=kernel_size//2
        # original conv
        self.conv_ori = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, padding=self.radius,
                                  stride=stride, dilation=dilation, groups=groups, padding_mode=padding_mode)
        # 8 different directions convs
        self.conv_l = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, self.radius+1), bias=bias, padding=0,
                                  stride=stride, dilation=dilation, groups=groups, padding_mode=padding_mode)
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, self.radius+1), bias=bias, padding=0,
                                  stride=stride, dilation=dilation, groups=groups, padding_mode=padding_mode)
        self.conv_u = nn.Conv2d(in_channels, out_channels, kernel_size=(self.radius+1, kernel_size), bias=bias, padding=0,
                                  stride=stride, dilation=dilation, groups=groups, padding_mode=padding_mode)
        self.conv_d = nn.Conv2d(in_channels, out_channels, kernel_size=(self.radius+1, kernel_size), bias=bias, padding=0,
                                  stride=stride, dilation=dilation, groups=groups, padding_mode=padding_mode)
        self.conv_nw = nn.Conv2d(in_channels, out_channels, kernel_size=(self.radius+1, self.radius+1), bias=bias, padding=0,
                                  stride=stride, dilation=dilation, groups=groups, padding_mode=padding_mode)
        self.conv_ne = nn.Conv2d(in_channels, out_channels, kernel_size=(self.radius+1, self.radius+1), bias=bias, padding=0,
                                  stride=stride, dilation=dilation, groups=groups, padding_mode=padding_mode)
        self.conv_sw = nn.Conv2d(in_channels, out_channels, kernel_size=(self.radius+1, self.radius+1), bias=bias, padding=0,
                                  stride=stride, dilation=dilation, groups=groups, padding_mode=padding_mode)
        self.conv_se = nn.Conv2d(in_channels, out_channels, kernel_size=(self.radius+1, self.radius+1), bias=bias, padding=0,
                                  stride=stride, dilation=dilation, groups=groups, padding_mode=padding_mode)

        self.conv_l.weight.data=self.conv_ori.weight.data[:, :, :, :self.radius+1]
        self.conv_r.weight.data=self.conv_ori.weight.data[:, :, :, self.radius:]
        self.conv_u.weight.data=self.conv_ori.weight.data[:, :, :self.radius+1, :]
        self.conv_d.weight.data=self.conv_ori.weight.data[:, :, self.radius:, :]
        self.conv_nw.weight.data=self.conv_ori.weight.data[:,:,:self.radius+1,:self.radius+1]
        self.conv_ne.weight.data=self.conv_ori.weight.data[:,:,:self.radius+1,self.radius:]
        self.conv_sw.weight.data=self.conv_ori.weight.data[:,:,self.radius:,:self.radius+1]
        self.conv_se.weight.data=self.conv_ori.weight.data[:,:,self.radius:,self.radius:]

    def forward(self, x):
        # custom padding    l       r      t             b
        # F.pad(,[lllll, rrrrr, ttttt, bbbbb])
        x_l = F.pad(x,[self.radius, 0, self.radius, self.radius])
        x_r = F.pad(x,[0, self.radius, self.radius, self.radius])
        x_u = F.pad(x,[self.radius, self.radius, self.radius, 0])
        x_d = F.pad(x,[self.radius, self.radius, 0, self.radius])
        x_nw = F.pad(x,[self.radius, 0, self.radius, 0])
        x_ne = F.pad(x,[self.radius, 0, 0, self.radius])
        x_sw = F.pad(x,[0, self.radius, self.radius, 0])
        x_se = F.pad(x,[0, self.radius, 0, self.radius])

        # 8 directions conv
        x_l = self.conv_l(x_l)
        x_r = self.conv_r(x_r)
        x_u = self.conv_u(x_u)
        x_d = self.conv_d(x_d)
        x_nw = self.conv_nw(x_nw)
        x_ne = self.conv_ne(x_ne)
        x_sw = self.conv_sw(x_sw)
        x_se = self.conv_se(x_se)

        # normalized 8 directions output
        x_l = x_l * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_l.weight.data)
        x_r = x_r * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_r.weight.data)
        x_u = x_u * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_u.weight.data)
        x_d = x_d * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_d.weight.data)
        x_nw = x_nw * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_nw.weight.data)
        x_ne = x_ne * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_ne.weight.data)
        x_sw = x_sw * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_sw.weight.data)
        x_se = x_se * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_se.weight.data)

        x_ori=self.conv_ori(x)

        return x_l,x_r,x_u,x_d,x_nw,x_ne,x_sw,x_se,x_ori
