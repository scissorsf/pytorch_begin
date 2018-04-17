import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable, Function

# x = torch.Tensor([[3,1,5],[-2,5,-7],[3,2,-4]])
# x =Variable(x, requires_grad =True)
# y = Variable(2*torch.ones(3,3))

# print("Input X:{}, \n Target Y:{} \n".format(x.data, y.data))
# print("Gradient of X:{}\n".format(x.grad))

#--------------------1-------------------
"""
Backpropagation ReLU
"""
# x = torch.Tensor([[3, 1, 5], [-2, 5, -7], [3, 2, -4]])

# x = Variable(x, requires_grad=True)
# y = Variable(2 * torch.ones(3, 3))

# print("Input X:{} \n Target Y:{}".format(x.data, y.data))

# out = F.relu(x)

# print("Output: {}".format(out.data))

# loss = torch.abs(out - y)
# print("Loss: {}".format(loss.data))
# loss = torch.sum(loss)
# loss.backward()

# print(x.grad)


#--------------------2-------------------
"""
Deconv ReLU
"""
# class DeconvReLU(Function):
#     @staticmethod
#     def forward(ctx, input_):
#         return input_.clamp(min=0)

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         print(" grad input :", grad_input)
#         grad_input[grad_input<0] = 0
#         return grad_input

# deconv_relu = DeconvReLU.apply


# x = torch.Tensor([[3, 1, 5], [-2, 5, -7], [3, 2, -4]])

# x = Variable(x, requires_grad=True)
# y = Variable(2 * torch.ones(3, 3))

# print("Input X:{}".format(x.data))

# out = deconv_relu(x)

# #print("Output: {}".format(out.data))

# loss = torch.abs(out - y)
# #print("Loss: {}".format(loss.data))
# loss = torch.sum(loss)
# loss.backward()

# print(x.grad)

#--------------------3-------------------
"""
Guided backpropagation ReLU
"""


class GuidedBackpropRelu(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return input_.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input_ = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        print("grad input: ", grad_input)
        grad_input[grad_input < 0] = 0
        grad_input[input_ < 0] = 0
        return grad_input


guided_backprop_relu = GuidedBackpropRelu.apply


x = torch.Tensor([[3, 1, 5], [-2, 5, -7], [3, 2, -4]])

x = Variable(x, requires_grad=True)
y = Variable(2 * torch.ones(3, 3))

print("Input X:{}".format(x.data))

out = guided_backprop_relu(x)

#print("Output: {}".format(out.data))

loss = torch.abs(out - y)
#print("Loss: {}".format(loss.data))
loss = torch.sum(loss)
loss.backward()

print(x.grad)
