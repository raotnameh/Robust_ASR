import torch
import torch.nn as nn

class Forget(torch.nn.Module):
  def __init__(self,in_channels):
    super(Forget, self).__init__()
    self.linear = torch.randn((in_channels, 1), requires_grad=True)
  def forward(self, x):
    print(self.linear)
    return self.linear*x
a = Forget(3)

print(a)
print(a.linear)
op = torch.optim.Adam([a.linear], lr=0.01,weight_decay=1e-4,amsgrad=True)
print(op)
print(op.param_groups)