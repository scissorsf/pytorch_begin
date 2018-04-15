# -*- coding: utf-8 -*-
"""
filename: explore_grad.py
Created on: 2018-4-13       
@author: Gfei
"""
import torch
from torch.autograd import Variable, profiler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class cacul_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = nn.Linear(3,1, bias=False)
        self.op2 = nn.Linear(1,1, bias = False)
        
    def forward(self, x):
        x_1 = self.op1(x)
        x_2 = self.op2(x_1)
        return x_1, x_2

def init_weights(m):
    #print(m)
    if type(m) == nn.Linear:
        m.weight.data.fill_(0.5)

inputs =Variable(torch.Tensor([2,4,6]))
out_target = Variable(torch.Tensor([32]))
print('Inputs x',inputs)

model = cacul_model()
#print('\nmodel_children:\n', list(model.children()))
#print('\nmodel_modules:\n', list(model.modules()))
print('\nmodel_named_children:\n', list(model.named_children()))
#print('\nmodel_named_modules:\n', list(model.named_modules()))


model.apply(init_weights)

optimizer = optim.SGD(model.parameters(), lr=0.001)
with profiler.profile() as prof:
    for steps in range(1):
        print('\n Traning times:  {}'.format(steps))
        optimizer.zero_grad()
        # for i, p in enumerate(model.parameters()):
        #     print('\n parameters:{}, value:{}'.format(i, p))
        #     print(p.grad)
        immediate_o,outputs = model(inputs)
        loss = F.mse_loss(outputs,out_target)
        print('\nloss:{}\n'.format(loss))
        print('\noutputs:{}\n'.format(type(outputs.grad_fn)))
        print(outputs.requires_grad)


        loss.backward()
        print(loss.grad_fn)
        print(loss.grad_fn.next_functions[0][0])
        

        for i,p in enumerate(model.parameters()):
            print('\n parameters:{}, value:{}'.format(i,p))
            print(p.grad)
        optimizer.step()
        # for i, p in enumerate(model.parameters()):
        #     print('\n parameters:{}, value:{}'.format(i, p))
        #for c in model.named_parameters():
        #    print('\n parameters:{}'.format(c))

        #print('\n parameters:{}'.format(model.state_dict()))

        print('\nResults:',immediate_o.data,outputs.data)
#print(prof)
