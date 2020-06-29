# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable
from .common import MLP, ResNet18
from .resnet import ResNet18 as ResNet18Full
import pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader

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
            self.net = MLP([784] + [128]*3 + [10])
            self.net.cuda()
            if args.data_file == 'notMNIST.pt':
                self.is_cifar = True
        
        self.lr = args.lr
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        
        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs
        # setup memories
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.n_memories = args.n_memories
        self.n_tasks = n_tasks
        if 'cub' in args.data_file:
            self.memx = torch.FloatTensor(n_tasks, self.n_memories, 3, 224, 224)
        elif 'mini' in args.data_file:
            self.memx = torch.FloatTensor(n_tasks, self.n_memories, 3, 84 ,84)
        else:
            self.memx = torch.FloatTensor(n_tasks, self.n_memories, n_inputs)
        self.memy = torch.LongTensor(n_tasks, self.n_memories)
        self.mem_feat = torch.FloatTensor(n_tasks, self.n_memories, self.nc_per_task)
        self.mem = {}
        if args.cuda:
            self.memx = self.memx.cuda()
            self.memy = self.memy.cuda()
            self.mem_feat = self.mem_feat.cuda()
        self.mem_cnt = 0
        self.n_memories = args.n_memories
        self.bsz = args.replay_batch_size
        
        self.n_outputs = n_outputs

        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss()
        self.samples_seen = 0
        self.samples_per_task = args.samples_per_task
        self.sz = args.replay_batch_size
        self.inner_steps = args.inner_steps
        self.beta = args.beta
        self.adapt_ = args.adapt
        self.adapt_lr = args.adapt_lr
        self.models = {}

    def on_epoch_end(self):  
        pass
    
    def adapt(self):
        print('Adapting')
        for t in range(self.n_tasks):
            model = deepcopy(self.net)
            if t > self.current_task:
                self.models[t] = model
                continue
            xx = self.memx[t]
            yy = self.memy[t]
            opt = torch.optim.SGD(model.parameters(), self.adapt_lr)
            train = torch.utils.data.TensorDataset(xx, yy)
            loader = DataLoader(train, batch_size = self.bsz, shuffle = True, num_workers =0)
            for _ in range(self.inner_steps):
                #for x,y in loader:
                model.zero_grad()
                pred = model.forward(xx)
                loss = self.bce(pred, yy)
                loss.backward()
                opt.step()
            self.models[t] = model

    def compute_offsets(self, task):
        if self.is_cifar:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)

    def forward(self, x, t, return_feat= False):
        if self.adapt_ and not self.net.training:
            output = self.models[t](x)
        else:
            output = self.net(x)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)

            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output
    
    def memory_sampling(self,t):
        mem_x = self.memx[:t,:]
        mem_y = self.memy[:t,:]
        mem_feat = self.mem_feat[:t,:]
        sz = min(self.sz, self.n_memories)
        idx = np.random.choice(t* self.n_memories,sz, False)
        t_idx = torch.from_numpy(idx // self.n_memories)
        s_idx = torch.from_numpy( idx % self.n_memories)

        offsets = torch.tensor([self.compute_offsets(i) for i in t_idx]).cuda()
        xx = mem_x[t_idx, s_idx]
        yy = mem_y[t_idx, s_idx] - offsets[:,0]
        feat = mem_feat[t_idx, s_idx]
        mask = torch.zeros(xx.size(0), self.nc_per_task)
        for j in range(mask.size(0)):
            mask[j] = torch.arange(offsets[j][0], offsets[j][1])
        return xx,yy, feat , mask.long().cuda()
    def observe(self, x, t, y):
        if t != self.current_task:
            tt = self.current_task
            offset1, offset2 = self.compute_offsets(tt)
            out = self.forward(self.memx[tt],tt, True)
            self.mem_feat[tt] = F.softmax(out[:, offset1:offset2] / self.temp, dim=1 ).data.clone()
            self.current_task = t
            self.mem_cnt = 0
        self.net.train()
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memx[t, self.mem_cnt: endcnt].copy_(x.data[: effbsz])
        self.memy[t, self.mem_cnt: endcnt].copy_(y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0
        
        weights_before = deepcopy(self.net.state_dict())
        for _ in range(self.inner_steps):
            self.net.zero_grad()
            loss1 = torch.tensor(0.).cuda()
            loss2 = torch.tensor(0.).cuda()
            loss3 = torch.tensor(0.).cuda()
         
            offset1, offset2 = self.compute_offsets(t)
            pred = self.forward(x,t, True)
            loss1 = self.bce(pred[:, offset1:offset2], y - offset1)
            if t > 0:
                xx, yy, target, mask = self.memory_sampling(t)
                pred_ = self.net(xx)
                pred = torch.gather(pred_, 1, mask)
                loss2 += self.reg* self.kl(F.log_softmax(pred / self.temp, dim =1 ), target)
                loss2 += self.bce(pred, yy)
                
            loss = loss1 + loss2
            loss.backward()
            self.opt.step()
        weights_after = self.net.state_dict()
        new_params = {name : weights_before[name] + ((weights_after[name] - weights_before[name]) * self.beta) for name in weights_before.keys()}
        self.net.load_state_dict(new_params)
        return loss.item()
