import torch
import torch.nn as nn

import sys
from math import pi

class sphere_loss(nn.Module):
    def __init__(self, m, scale, num_class, use_gpu):
        super(sphere_loss, self).__init__()
        self.m = m
        self.scale = scale
        self.num_class = num_class
        self.use_gpu = use_gpu
        self.loss = nn.CrossEntropyLoss()

    def theta_to_psi(self, theta_yi_i):
        k = torch.floor(theta_yi_i * self.m / pi)
        sign = torch.full(k.shape, -1)
        if self.use_gpu:
            sign = sign.cuda()
        co = torch.pow(sign, k)
        cos_m_theta_yi_i = torch.cos(self.m * theta_yi_i)
        return co * cos_m_theta_yi_i - 2 * k

    def forward(self, y_hat, y):
        y = torch.unsqueeze(y, 0)
        label = torch.reshape(y, (y.shape[1], 1))
        one_hot = torch.zeros(y.shape[1], self.num_class)
        if self.use_gpu:
            one_hot = one_hot.cuda()
        one_hot = one_hot.scatter_(1, label, 1)
        mask = one_hot.to(torch.bool)
        #theta(yi, i)
        cos_theta_yi_i = torch.masked_select(y_hat, mask)
        theta_yi_i = torch.acos(cos_theta_yi_i)
        psi_yi_i = self.theta_to_psi(theta_yi_i)

        fc = y_hat * 1.0
        index = torch.Tensor(range(y_hat.shape[0]))
        fc[index.long(), y.long()] = psi_yi_i[index.long()]
        fc = fc * self.scale

        y = y.squeeze(0)
        loss = self.loss(fc, y)

        return loss
