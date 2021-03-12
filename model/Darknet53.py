import torch
import torch.utils.tensorboard
import numpy as np

from torch import nn


def DBL(in_c, out_c, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU())


class Res_unit(nn.Module):
    def __init__(self, in_c):
        super(Res_unit, self).__init__()

        reduce_c = int(in_c / 2)
        self.layer1 = DBL(in_c, reduce_c, kernel_size=1, padding=0)
        self.layer2 = DBL(reduce_c, in_c, kernel_size=3)

    def forward(self, x):
        input = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += input
        return out


class darknet53(nn.Module):
    def __init__(self):
        super(darknet53, self).__init__()

        self.conv1 = DBL(in_c=3, out_c=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = DBL(in_c=32, out_c=64, kernel_size=3, stride=2, padding=1)
        self.res1 = self.residual_box(in_c=64, in_num=1)
        self.conv3 = DBL(in_c=64, out_c=128, kernel_size=3, stride=2, padding=1)
        self.res2 = self.residual_box(in_c=128, in_num=2)
        self.conv4 = DBL(in_c=128, out_c=256, kernel_size=3, stride=2, padding=1)
        self.res3 = self.residual_box(in_c=256, in_num=8)

        self.conv5 = DBL(in_c=256, out_c=512, kernel_size=3, stride=2, padding=1)
        self.res4 = self.residual_box(in_c=512, in_num=8)
        self.conv6 = DBL(in_c=512, out_c=1024, kernel_size=3, stride=2, padding=1)
        self.res5 = self.residual_box(in_c=1024, in_num=4)

    def residual_box(self, in_c, in_num):
        #         half_c = in_c // 2
        # DBL(in_c, half_c, 1, 0 )
        for i in range(in_num):
            box = Res_unit(in_c)
        return box

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out)
        out = self.conv3(out)
        out = self.res2(out)
        out = self.conv4(out)
        r1 = self.res3(out)
        out = self.conv5(r1)
        r2 = self.res4(out)
        out = self.conv6(r2)
        r3 = self.res5(out)
        # print("r3_shape:",r3.shape)
        return r1, r2, r3

# img = torch.randn([1,3,416,416])
# darknet=darknet53()
# q= darknet.forward(img)
# print(d[0].shape)
# print(d[1].shape)
# print(d[2].shape)

