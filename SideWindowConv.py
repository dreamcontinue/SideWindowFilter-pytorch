import torch
import torch.nn as nn
import torch.nn.functional as F


class SideWindowConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=False):
        super(SideWindowConv, self).__init__()

        assert kernel_size >= 3, 'kernel size must great than 3'
        self.radius = kernel_size//2
        # Original conv to get 3x3 filter parameters
        self.conv_ori = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, padding=self.radius,
                                  stride=stride, dilation=dilation, groups=groups)
        # 8 different directions side window convs
        self.conv_l = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, self.radius+1), bias=bias,
                                padding=0, stride=stride, dilation=dilation, groups=groups)
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, self.radius+1), bias=bias,
                                padding=0, stride=stride, dilation=dilation, groups=groups)
        self.conv_u = nn.Conv2d(in_channels, out_channels, kernel_size=(self.radius+1, kernel_size), bias=bias,
                                padding=0, stride=stride, dilation=dilation, groups=groups)
        self.conv_d = nn.Conv2d(in_channels, out_channels, kernel_size=(self.radius+1, kernel_size), bias=bias,
                                padding=0, stride=stride, dilation=dilation, groups=groups)
        self.conv_nw = nn.Conv2d(in_channels, out_channels, kernel_size=(self.radius+1, self.radius+1), bias=bias,
                                padding=0, stride=stride, dilation=dilation, groups=groups)
        self.conv_ne = nn.Conv2d(in_channels, out_channels, kernel_size=(self.radius+1, self.radius+1), bias=bias,
                                padding=0, stride=stride, dilation=dilation, groups=groups)
        self.conv_sw = nn.Conv2d(in_channels, out_channels, kernel_size=(self.radius+1, self.radius+1), bias=bias,
                                padding=0, stride=stride, dilation=dilation, groups=groups)
        self.conv_se = nn.Conv2d(in_channels, out_channels, kernel_size=(self.radius+1, self.radius+1), bias=bias,
                                padding=0, stride=stride, dilation=dilation, groups=groups)
        # Set filters
        self.conv_l.weight.data=self.conv_ori.weight.data[:, :, :, :self.radius+1]
        self.conv_r.weight.data=self.conv_ori.weight.data[:, :, :, self.radius:]
        self.conv_u.weight.data=self.conv_ori.weight.data[:, :, :self.radius+1, :]
        self.conv_d.weight.data=self.conv_ori.weight.data[:, :, self.radius:, :]
        self.conv_nw.weight.data=self.conv_ori.weight.data[:,:,:self.radius+1,:self.radius+1]
        self.conv_ne.weight.data=self.conv_ori.weight.data[:,:,:self.radius+1,self.radius:]
        self.conv_sw.weight.data=self.conv_ori.weight.data[:,:,self.radius:,:self.radius+1]
        self.conv_se.weight.data=self.conv_ori.weight.data[:,:,self.radius:,self.radius:]


    def forward(self, x):
        # Custom padding    l       r      t             b
        # F.pad(,[lllll, rrrrr, ttttt, bbbbb])
        x_l = F.pad(x, [self.radius, 0, self.radius, self.radius])
        x_r = F.pad(x, [0, self.radius, self.radius, self.radius])
        x_u = F.pad(x, [self.radius, self.radius, self.radius, 0])
        x_d = F.pad(x, [self.radius, self.radius, 0, self.radius])
        x_nw = F.pad(x, [self.radius, 0, self.radius, 0])
        x_ne = F.pad(x, [self.radius, 0, 0, self.radius])
        x_sw = F.pad(x, [0, self.radius, self.radius, 0])
        x_se = F.pad(x, [0, self.radius, 0, self.radius])

        # 8 directions side window conv
        x_l = self.conv_l(x_l)
        x_r = self.conv_r(x_r)
        x_u = self.conv_u(x_u)
        x_d = self.conv_d(x_d)
        x_nw = self.conv_nw(x_nw)
        x_ne = self.conv_ne(x_ne)
        x_sw = self.conv_sw(x_sw)
        x_se = self.conv_se(x_se)

        # Normalized output
        x_l = x_l * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_l.weight.data) - x
        x_r = x_r * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_r.weight.data) - x
        x_u = x_u * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_u.weight.data) - x
        x_d = x_d * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_d.weight.data) - x
        x_nw = x_nw * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_nw.weight.data) - x
        x_ne = x_ne * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_ne.weight.data) - x
        x_sw = x_sw * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_sw.weight.data) - x
        x_se = x_se * torch.sum(self.conv_ori.weight.data) / torch.sum(self.conv_se.weight.data) - x

        # Take the min
        x_mutidim = torch.stack((x_l, x_r, x_u, x_d, x_nw, x_ne, x_sw, x_se), dim=2)
        b, c, d, h, w = x_mutidim.shape
        x_mutidim_min_mask = torch.argmin(torch.abs(x_mutidim), dim=2, keepdim=True)
        x_mutidim = torch.gather(x_mutidim, dim=2, index=x_mutidim_min_mask)
        x_mutidim = x_mutidim.view(b,c,h,w)

        return x_mutidim + x
