import torch
from torch.profiler import profile, record_function, ProfilerActivity
from arnoldi_and_lanczos_iterations import *


# torch.set_printoptions(precision=10)
n = 25
scalar = 100
A = torch.randn(n, n)*scalar
# symmetric A:
A = A + torch.t(A)
eigvals = torch.linalg.eigvals(A)
# print(eigvals)

b = torch.randn(n)
functions = [arnoldi_iteration_modified]
size = len(functions)
i = 0
for f in functions:
    i += 1
    with profile(activities=[ProfilerActivity.CPU],
        profile_memory=True, record_shapes=True) as prof:
        V, H = f(A, b, n)
    eigvals_aprox = torch.linalg.eigvals(H)
    # print(H)
    nEigvals = len(eigvals_aprox)
    errors = torch.zeros(nEigvals)
    for i in range(nEigvals):
        error = min(abs(eigvals-eigvals_aprox[i]))
        errors[i] = error
    print(errors)
    prof.export_chrome_trace("trace.json")


# n = 50
# A = torch.randn(n, n)
# eigvals, eigvecs = torch.linalg.eig(A)
# b = torch.randn(n)
# b = b/torch.linalg.norm(b)
# m = n
# V, H = arnoldi_iteration(A, b, m)
# eigvals_aprox, eigvecs_aprox = torch.linalg.eig(H)
# print("Approximated Eigenvalues with no Dimension Reduction:")
# eigvals = torch.sort(eigvals)
# eigvals_aprox = torch.sort(eigvals_aprox)
# print(abs(eigvals-eigvals_aprox))
