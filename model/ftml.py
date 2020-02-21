# An implementation of MER Algorithm 1 from https://openreview.net/pdf?id=B1gTShAct7

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pdb
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from .common import MLP, ResNet18, CNN
from .resnet import ResNet18 as ResNet18Full
import random
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from random import shuffle
import sys
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        
        self.is_cifar = any(x in str(args.data_file) for x in ['cifar', 'cub', 'mini'])
        if 'cifar' in args.data_file or 'mini' in args.data_file:
            self.net = ResNet18(n_outputs)
        elif 'cub' in args.data_file:
            self.net = ResNet18Full(args.pretrained, n_outputs)
        else:
            self.net = MLP([784] + [128]*3 + [10])
        
        self.bce = CrossEntropyLoss()

        self.n_meta = args.n_meta
        
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.opt = optim.SGD(self.parameters(), args.lr)
        self.age = 0

        self.bsz = args.replay_batch_size
        self.lr = args.lr
        self.adapt_ = args.adapt
        self.grad_step = args.inner_steps
        if self.is_cifar:
            self.nc_per_task = n_outputs/n_tasks
        else:
            self.nc_per_task = n_outputs

        self.memories = args.n_memories 
        self.steps = int(args.batches_per_example)
        self.beta = args.beta
        self.gamma = args.gamma
        
        # allocate buffer
        self.n_memories = args.n_memories
        if 'cub' in args.data_file:
            self.memx = torch.FloatTensor(n_tasks, self.n_memories, 3, 224, 224)
        elif 'mini' in args.data_file:
            self.memx = torch.FloatTensor(n_tasks, self.n_memories, 3, 128 ,128)
        else:
            self.memx = torch.FloatTensor(n_tasks, self.n_memories, n_inputs)
        self.memy = torch.LongTensor(n_tasks, self.n_memories)
        
        if args.cuda:
            self.memx = self.memx.cuda()
            self.memy = self.memy.cuda()
        
        self.mem_cnt= 0
        self.current_task = 0
        self.models = {}

        # handle gpus if specified
        self.cuda = args.cuda
        if self.cuda:
            self.net = self.net.cuda()

        # meta params
        self.beta = args.beta
        self.adapt_lr = args.adapt_lr

    def compute_offsets(self, task):
        if self.is_cifar:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return (int(offset1), int(offset2))

    def forward(self, x, t):
        '''
        if self.net.training:
            output = self.net(x)
        else:
            output = self.models[t](x)
        '''
        offset1, offset2 = self.compute_offsets(t)

        if self.adapt_ and not self.net.training:
            output = self.models[t](x)
        else:
            output = self.net(x)
        if self.is_cifar:
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output

    def on_epoch_end(self):
        pass

    def getBatch(self, t):
        #data = [x for x in self.M if x[-1] == t]
        if self.net.training and t == self.current_task:
            xx = self.memx[t,:self.age]
            yy = self.memy[t,:self.age]
        else:
            xx = self.memx[t,:]
            yy = self.memy[t,:]
        return xx, yy

    def adapt(self):
        print('Adapting')
        for t in range(self.n_tasks):
            model = deepcopy(self.net)
            opt = optim.SGD(model.parameters(), self.adapt_lr)
            # data prepare  
            xx, yy = self.getBatch(t)
            if t > self.current_task:
                self.models[t] = model
                continue
            train = torch.utils.data.TensorDataset(xx, yy)
            loader = DataLoader(train, batch_size = self.bsz, shuffle = True, num_workers =0)
            
            for _ in range(self.grad_step): 
                #for x, y in loader:
                model.zero_grad()
                pred = model.forward(xx)
                loss = self.bce(pred, yy)
                loss.backward()
                opt.step()
                
            self.models[t] = model


    def observe(self, x, t, y):
        if t!= self.current_task:
            self.current_task = t
            self.mem_cnt = 0
            self.age = 0
        
        self.net.train()
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memx[t, self.mem_cnt: endcnt].copy_(x.data[: effbsz])
        self.memy[t, self.mem_cnt: endcnt].copy_(y.data[: effbsz])
        self.mem_cnt += effbsz
        self.age += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0
            self.age = self.n_memories
        
        weights_before = deepcopy(self.net.state_dict())
        for _ in range(self.grad_step):
            self.net.zero_grad()
            pred = self.forward(x, t)
            offset1, offset2 = self.compute_offsets(t)
            pred = pred[:,offset1:offset2]
            loss = self.bce(pred, y-offset1)
            loss.backward()
            self.opt.step()
        weights_after = self.net.state_dict()
        
        new_params = {name : weights_before[name] + ((weights_after[name] - weights_before[name]) * self.beta) for name in weights_before.keys()}
        self.net.load_state_dict(new_params) 
        # meta Update
        for _ in range(self.n_meta):
            idx = np.random.choice(t+1)
            xx , yy = self.getBatch(idx)
            weights_before = deepcopy(self.net.state_dict())
            beta = {k: self.beta for k in weights_before.keys()}
            for tmp in range(self.grad_step):
                self.net.zero_grad()
                pred = self.forward(xx,idx)
                offset1, offset2 = self.compute_offsets(idx)
                pred = pred[:,offset1:offset2]
                loss = self.bce(pred, yy - offset1)
                loss.backward()
                self.opt.step()
            weights_after = self.net.state_dict()
            new_params = {name : weights_before[name] + ((weights_after[name] - weights_before[name]) * beta[name]) for name in weights_before.keys()}
            self.net.load_state_dict(new_params)  
        return loss
