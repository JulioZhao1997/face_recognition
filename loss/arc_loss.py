import torch
import torch.nn as nn

import math
import sys

class arc_loss(nn.Module):
    def __init__(self, num_class, margin, scale, use_gpu):
        super(arc_loss, self).__init__()
        self.num_class = num_class
        self.margin = margin
        self.scale = scale
        self.loss = nn.CrossEntropyLoss()
        self.use_gpu = use_gpu

    def forward(self, y_hat, y):
        fc = y_hat * 1.0
        y = y.unsqueeze(0)
        if self.use_gpu:
            y, y_hat = y.cuda(), y_hat.cuda()
        label = torch.reshape(y, (y.shape[1], 1))
        one_hot = torch.zeros(y.shape[1], self.num_class)
        if self.use_gpu:
            one_hot = one_hot.cuda()
        one_hot = one_hot.scatter_(1, label, 1)
        mask = one_hot.to(torch.bool)
        cos_theta_yi = torch.masked_select(y_hat, mask)
        theta_yi = torch.acos(cos_theta_yi)

        #add margin
        theta_yi_m = theta_yi + self.margin
		
        #limit 0 < (theta + m) < pi
        overflow_mask = theta_yi_m > math.pi
        if self.use_gpu:
            overflow_index = torch.masked_select(torch.arange(y_hat.shape[0], dtype=torch.long).cuda(), overflow_mask)
        else:
            overflow_index = torch.masked_select(torch.arange(y_hat.shape[0], dtype=torch.long), overflow_mask)
        theta_yi_m[overflow_index] = theta_yi[overflow_index]
        cos_theta_yi_m = torch.cos(theta_yi_m)
        index  = torch.Tensor(range(y_hat.shape[0]))
        fc[index.long(), y.long()] = cos_theta_yi_m[index.long()]
        fc = fc * self.scale

        y = y.squeeze(0)
        loss = self.loss(fc, y)
        return loss
