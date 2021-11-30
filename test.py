import torch

a = torch.tensor([3])
v = torch.tensor([[1], [2]])
print(v)
w = torch.t(v)
print(a@v)
