# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable
from .common import MLP, ResNet18, CNN
from .resnet import ResNet18 as ResNet18Full
import pdb
import torch.nn as nn
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.reg = args.memory_strength
        self.temp = args.temperature
        # setup network
        self.is_cifar = any(x in str(args.data_file) for x in ['cifar', 'cub', 'mini'])
        if 'cifar' in args.data_file or 'mini' in args.data_file:
            self.net = ResNet18(n_outputs)
        elif 'cub' in args.data_file:
            self.net = ResNet18Full(True, n_outputs)
        else:
            #self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])
            self.net = MLP([784] + [128]*3 + [10])
            if args.data_file == 'notMNIST.pt':
                self.is_cifar = True
        # setup optimizer
        self.lr = args.lr
        #if self.is_cifar:
        #    self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        #else:
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        
        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        # setup memories
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.n_memories = args.n_memories
        
        #self.memx = torch.FloatTensor(n_tasks, self.n_memories, n_inputs)
        #self.memy = torch.FloatTensor(n_tasks, self.n_memories, n_outputs)
        self.mem = {}

        self.mem_cnt = 0
        self.n_memories = args.n_memories

        if self.is_cifar:
            self.nc_per_task = n_outputs / n_tasks
        else:
            self.nc_per_task = n_outputs
        self.n_outputs = n_outputs

        self.kl = nn.KLDivLoss()
        self.samples_seen = 0
        self.samples_per_task = args.samples_per_task
        self.reg_keys = True
    def reset_optim(self):
        #if self.is_cifar:
        #    self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        #else:
        self.reg_keys = False

    def compute_offsets(self, task):
        if self.is_cifar:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)
    
    def on_epoch_end(self):
        self.reg_keys = False

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, info, y):
        t = info[0]
        idx = info[1]
        if t != self.current_task:
            self.current_task = t
            self.samples_seen = 0
            self.mem = {}
            self.reg_keys = True

        if self.reg_keys:
            if t == 0: return 0
            tt = t
            offset1, offset2 = self.compute_offsets(tt)
            pred = self.net(x)[:,offset1:offset2]
            for j in range(x.size(0)):
                key = idx[j].item()
                self.mem[key] = pred[j,:].data
            return 0
        self.net.zero_grad()
        loss1 = torch.tensor(0.).cuda()
        loss2 = torch.tensor(0.).cuda()
        loss3 = torch.tensor(0.).cuda()
        
        offset1, offset2 = self.compute_offsets(t)
        pred = self.net(x)
        loss1 = self.bce((pred[:, offset1: offset2]), y - offset1)

        if t > 0:
            tt = t
            offset1, offset2 = self.compute_offsets(tt)
            key_y = None
            for j in range(x.size(0)):
                x_ = x[j,:]
                key_ = self.mem[idx[j].item()]
                if key_y is None:
                    key_y = key_.unsqueeze(0)
                else:
                    key_y = torch.cat([key_y, key_.unsqueeze(0)], dim = 0)
            
            prev_pred = pred[:, offset1:offset2]
            loss2 += self.reg * self.kl(F.log_softmax(prev_pred / self.temp, dim=1), key_y)
        loss = loss1 + loss2
        loss.backward()
        self.opt.step()

        return loss.item()
