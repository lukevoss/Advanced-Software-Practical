import torch
from arnoldi_and_lanczos_iterations import *
import tracemalloc


# torch.set_printoptions(precision=10)
n = 10
A = torch.randn(n, n)*100
# symmetric A:
A = A + torch.t(A)
eigvals = torch.linalg.eigvals(A)

# print(eigvals)

b = torch.randn(n)
functions = [lanczos_iteration_saad, arnoldi_iteration]
size = len(functions)
i = 0
for f in functions:
    i += 1
    tracemalloc.start()
    #tracemalloc.reset_peak()
    V, H = f(A, b, n)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current: {current} Bytes, Peak: {peak} Bytes")
    tracemalloc.stop()
    tracemalloc.clear_traces()

    eigvals_aprox = torch.linalg.eigvals(H)
    # print(H)
    nEigvals = len(eigvals_aprox)
    errors = torch.zeros(nEigvals)
    for i in range(nEigvals):
        error = min(abs(eigvals-eigvals_aprox[i]))
        errors[i] = error
    # print(errors)
    # print(H.dtype)


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
