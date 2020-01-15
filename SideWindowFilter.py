import torch
import torch.nn as nn
import torch.nn.functional as F

class SideWindowFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=False):
        super(SideWindowFilter, self).__init__()

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

    def set_filter(self, filter=None):
        if filter is not None:
            self.conv_ori.weight.data = filter
        self.conv_l.weight.data = self.conv_ori.weight.data[:, :, :, :self.radius+1]
        self.conv_r.weight.data = self.conv_ori.weight.data[:, :, :, self.radius:]
        self.conv_u.weight.data = self.conv_ori.weight.data[:, :, :self.radius+1, :]
        self.conv_d.weight.data = self.conv_ori.weight.data[:, :, self.radius:, :]
        self.conv_nw.weight.data = self.conv_ori.weight.data[:,:,:self.radius+1,:self.radius+1]
        self.conv_ne.weight.data = self.conv_ori.weight.data[:,:,:self.radius+1,self.radius:]
        self.conv_sw.weight.data = self.conv_ori.weight.data[:,:,self.radius:,:self.radius+1]
        self.conv_se.weight.data = self.conv_ori.weight.data[:,:,self.radius:,self.radius:]

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


import numpy as np
from PIL import Image
from torch.autograd import Variable

def preprocess_image(pil_im):
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
    im_as_ten = torch.from_numpy(im_as_arr).float()
    im_as_ten.unsqueeze_(0)
    im_as_var = Variable(im_as_ten, requires_grad=False)
    with torch.no_grad():
        im_as_var=torch.autograd.Variable(im_as_var)
    return im_as_var

def recover_image(im_as_var):
    im_as_var=im_as_var.squeeze_()
    img=im_as_var.detach().numpy()
    for channel, _ in enumerate(img):
        img[channel] *= 255
    img=img.transpose(1,2,0)
    img=Image.fromarray(np.uint8(img),'RGB')
    return img

def sidewindowfilter(img,swf):
    b,c,w,h=img.shape
    for i in range(c):
        img[:,i,:,:] = swf(img[:,i,:,:].view(b,1,w,h)).view(b,w,h)
    return img

def process(img_path,filter,iteration=1):
    img = Image.open(img_path).convert('RGB')
    # img.show()

    img_as_var = preprocess_image(img)
    if filter is not None:
        swf = SideWindowFilter(1, 1, filter.shape[-1])
        filter = torch.from_numpy(filter).float()
        swf.set_filter(filter)
    else:
        swf = SideWindowFilter(1, 1, 3)
    for i in range(iteration):
        img_as_var = sidewindowfilter(img_as_var, swf)

    img = recover_image(img_as_var)
    # img.show()
    return img

if __name__ == '__main__':
    img_path = 'img/ori.jpg'
    filter = np.array([[[[0.0453542,0.0566406,0.0453542],
                       [0.0566406,0.0707355,0.0566406],
                       [0.0453542,0.0566406,0.0453542]]]], dtype=np.float)
    # filter=np.array([[[[1,1,1,1,1],
    #                    [1,1,1,1,1],
    #                    [1,1,1,1,1],
    #                    [1,1,1,1,1],
    #                    [1,1,1,1,1]]]], dtype=np.float)
    filter /= np.sum(filter)
    img = process(img_path, filter=filter, iteration=1)
    # img.save('img/process.jpg')

