import torch
from arnoldi_and_lanczos_iterations import *

# n=5
# A = torch.randn(n, n)
# A_symm = A + torch.t(A)
# print(A_symm)
# b = torch.randn(n)
# functions = [arnoldi_iteration, lanczos_iteration]
# size = len(functions)
# print(size)
# i = 0
# for f in functions:
#     i += 1
#     V, H = f(A_symm, b, 5)
#     print(H)

temp = torch.zeros(2,5)
temp[0,1] = 6
print (temp[0,:])