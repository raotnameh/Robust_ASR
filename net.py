# import torch
# import torch.nn as nn

# class Forget(torch.nn.Module):
#   def __init__(self,in_channels):
#     super(Forget, self).__init__()
#     self.linear = torch.nn.Parameter(torch.randn((in_channels, 1), requires_grad=True))
#   def forward(self, x):
#     print(self.linear)
#     return self.linear*x
# a = Forget(3)

# print(a)
# # print(a.linear)
# # print(a.named_parameters)
# # op = torch.optim.Adam([a.linear], lr=0.01,weight_decay=1e-4,amsgrad=True)
# # print(op)
# # print(op.param_groups)


import random
import torch
import math


class DynamicNet(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 4, 5
        and reuse the e parameter to compute the contribution of these orders.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same parameter many
        times when defining a computational graph.
        """
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x ** exp
        return y



model = DynamicNet()
print(model)

print(model.parameters())

print(model.a)