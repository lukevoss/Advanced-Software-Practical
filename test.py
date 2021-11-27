import torch

A = torch.tensor([[4, 1], [2, 3]])
v = torch.tensor([[1], [2]])
X = A @ v
print(X)
for i in range(5):
    print(i)
