import torch

from torch.autograd import Variable
from torch.autograd import Function

class MyReLU(Function):
    
    def forward(self, input_):
        self.save_for_backward(input_)
        output = input_.clamp(min=0)
        return output
    
    def backward(self, grad_output):
        input_,  = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_< 0] = 0
        return grad_input


i_ = Variable(torch.randn(2), requires_grad=True)

relu = MyReLU()

o_1 = relu(i_)
o_ = torch.pow(o_1,2)

o_1.register_hook(print)
print(i_, ' -> ', o_)
print(o_.grad_fn)
print(o_.grad_fn.next_functions[0][0])
print(o_.grad_fn.next_functions[0][0].next_functions)

o_.backward(torch.ones(2))
print(i_.grad)
# Only leaf Variables's grad is retain for saving memory!
print(o_1.grad)
print(o_.grad)
