import torch 
import torch.nn as nn
from torch.autograd import Variable

batch_size = 4

"""
Define Class
-- Define Model
-- Set Weights to 1 and Bias to 0
"""

class Linear(nn.Module):
    
    def __init__(self, feature_list):
        super().__init__()
        self.feature_list = feature_list
        self.layers = []
        self.h = []

        for i in range(len(feature_list)-1):
            self.layers.append(nn.Linear(self.feature_list[i],self.feature_list[i+1]))
        self.total = nn.ModuleList(self.layers)

        for idx,m in enumerate(self.total):
            if isinstance(m, nn.Linear):
                # m.weight.data.fill_(1)
                m.bias.data.fill_(0)
            if idx == 0 or 1 :
                # self.h0 = m.register_forward_hook(self.forward_hook)
                # self.h1 = m.register_forward_pre_hook(self.forward_pre_hook)
                self.h.append(m.register_backward_hook(self.backward_hook))

    def forward_hook(self, *args):
        module, input_, output_ = args[0],args[1],args[2]
        print("\n This is Forward Hook. \n")
        for i in args:
            print(i)
        
    def backward_hook(self, *args):
        module, grad_input, grad_output = args[0], args[1], args[2]
        print("\n This is Backward Hook. \n")
        for i in args:
            print(i)

    def forward_pre_hook(self, *args):
        module, input_ = args[0], args[1]
        print("\n This is Forward Pre Hook. \n")
        for i in args:
            print(i)

    def remove_hook(self):
        # self.h0.remove()
        # self.h1.remove()
        for i in self.h:
            i.remove()
    
    def forward(self,x):
        out = x
        for idx, layer in enumerate(self.total):
            out = layer(out)
        return out

feature_list =[1,2,4]
model = Linear(feature_list)
print(model)

x = Variable(torch.ones(batch_size,1), requires_grad = True)
out = model(x)
print(out)

# out = torch.sum(out)
out.backward(torch.ones((batch_size,4)))

model.remove_hook()

# x = Variable(torch.ones(batch_size,1), requires_grad=True)
# out = model(x)
# out = torch.sum(out)
# out.backward()
