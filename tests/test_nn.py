
import numpy as np
import torch


def make_labels(bs, val):
  y = np.zeros((bs, 2), np.float32)
  y[range(bs), [val] * bs] = 1  # Can we do label smoothin? i.e -2.0 changed to -1.98789.
  y[range(bs), [1-val] * bs] = 0
  return torch.Tensor(y)




x = torch.randn((8, 2))


loss_func = torch.nn.CrossEntropyLoss()

print(make_labels(8, 0))



print(x.log_softmax(dim=-1))

x1 = torch.Tensor([3])
x2 = torch.Tensor([5])

print((x1 + x2).mean())
# import torch
# import torch.nn as nn
# import numpy as np

# a = np.arange(1,13).reshape(3,4)
# b = torch.from_numpy(a)
# x_input = b.float()
# print('input:\n',x_input)

# y_target = torch.tensor([1,2,3])
# print('y_target:\n',y_target)

# crossentropyloss=nn.CrossEntropyLoss(reduction='none')
# crossentropyloss_output=crossentropyloss(x_input,y_target)
# print('crossentropyloss_output:\n',crossentropyloss_output)


# softmax_func=nn.Softmax(dim=1)
# soft_output=softmax_func(x_input)
# print('soft_output:\n',soft_output)

# log_output=torch.log(soft_output)
# print('log_output:\n',log_output)

# nllloss_func=nn.NLLLoss(reduction='none')
# nllloss_output=nllloss_func(log_output, y_target)
# print('nllloss_output:\n',nllloss_output)