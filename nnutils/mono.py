from __future__ import print_function
import pdb
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import torchvision.models as models

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.LeakyReLU(0.1, inplace=True),)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.LeakyReLU(0.1, inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class pyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes,with_bn=True):
        super(pyramidPooling, self).__init__()

        bias = not with_bn

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=bias, with_bn=with_bn))

        self.conv_last = conv2DBatchNormRelu(in_channels*2, in_channels, 1, 1, 0, bias=bias, with_bn=with_bn)
    
        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
    
    def forward(self, x):
        h, w = x.shape[2:]

        k_sizes = []
        strides = []
        k_sizes = [(self.pool_sizes[0],self.pool_sizes[0]),(self.pool_sizes[1],self.pool_sizes[1]) ,(self.pool_sizes[2],self.pool_sizes[2]) ,(self.pool_sizes[3],self.pool_sizes[3])]
        strides = k_sizes

        output_slices = [x]

        for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
            out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
            out = module(out)
            out = F.upsample(out, size=(h,w), mode='bilinear')
            output_slices.append(out)

        return self.conv_last(torch.cat(output_slices, dim=1))



class mono(nn.Module):
    def __init__(self):
        super(mono, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.conv1 = nn.Sequential(*[resnet.conv1, resnet.bn1,resnet.relu])
        self.pool1 = resnet.maxpool
        self.conv2 = resnet.layer1
        self.conv3 = resnet.layer2
        self.conv4 = resnet.layer3
        self.conv5 = resnet.layer4

        self.pyramid_pooling = pyramidPooling(512, [1, 2, 4, 8])
        self.upconv5 = nn.Sequential(*[nn.Upsample(scale_factor=2), conv2DBatchNormRelu(512, 256, 3, 1, 1, bias=False, with_bn=True)])
        self.iconv5 = conv2DBatchNormRelu(512, 256, 3, 1, 1, bias=False, with_bn=True)
        self.upconv4 = nn.Sequential(*[nn.Upsample(scale_factor=2), conv2DBatchNormRelu(256, 128, 3, 1, 1, bias=False, with_bn=True)])
        self.iconv4 = conv2DBatchNormRelu(256, 128, 3, 1, 1, bias=False, with_bn=True)
        self.upconv3 = nn.Sequential(*[nn.Upsample(scale_factor=2), conv2DBatchNormRelu(128, 64, 3, 1, 1, bias=False, with_bn=True)])
        self.iconv3 = conv2DBatchNormRelu(128, 64, 3, 1, 1, bias=False, with_bn=True)
        self.pred = nn.Conv2d(64, 1, kernel_size=3,padding=1, stride=1, bias=True)

    def forward(self, left):
        conv1 = self.conv1(left)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        conv5_spp = self.pyramid_pooling(conv5)
        
        upconv5 = self.upconv5(conv5_spp) #H/16
        concat5 = torch.cat([upconv5, conv4], 1)
        iconv5  = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5) #h/8
        concat4 = torch.cat([upconv4, conv3], 1)
        iconv4  = self.iconv4(concat4)

        upconv3 = self.upconv3(iconv4) #h/4
        concat3 = torch.cat([upconv3, conv2], 1)
        iconv3  = self.iconv3(concat3)

        pred_q = self.pred(iconv3)
        pred = F.upsample(pred_q, scale_factor=4,mode='bilinear') 

        return pred
