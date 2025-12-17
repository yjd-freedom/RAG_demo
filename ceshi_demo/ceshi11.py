import random

import torch
torch.manual_seed(1)
data = torch.rand(2,3)
print(data)
print(data.shape)
data1 = data[[1]]
print(data1,data1.shape)
# a = torch.tensor()
data2 = data[1]
print(data2,data2.shape)
print(data[0],data[0].shape)