import torch
from arnoldi_and_lanczos_iterations import *
# create MAtrix A with known Eigenvalues 1 to n=25
n = 30
eigvals = torch.linspace(1., n, n)
eigvecs = torch.randn(n, n)
A = torch.linalg.solve(eigvecs, (torch.diag(eigvals) @ eigvecs))
# double check if created Matrix A has Eigenvalues 1 to 25
eigvals_test, eigvecs_test = torch.linalg.eig(A)
print("Eigenvalues of created Matrix:")
print(eigvals_test)

# Picking a starting vector and then normalizing it
b = torch.randn(n)
b = b/torch.linalg.norm(b)

# Testing Arnoldi algorithm with m=30
m = 30
V, H = arnoldi_iteration(A, b, m)
eigvals_aprox, eigvecs_aprox = torch.linalg.eig(H)
print("Approximated Eigenvalues with no Dimension Reduction:")
print(eigvals_aprox)



# Testing Arnoldi algorithm with m=10
m = 10
V, H = arnoldi_iteration(A, b, m)
eigvals_aprox, eigvecs_aprox = torch.linalg.eig(H)
print("Approximated Eigenvalues with Dimension Reduction:")
print(eigvals_aprox)

# Testing if V^(T)*A*V=H
norm_VAV = torch.linalg.norm(torch.t(V) @ A @ V)
norm_H = torch.linalg.norm(H)
print("Test of two norms")
print(norm_VAV)
print(norm_H)
print("Error of norm:")
print(abs(norm_VAV-norm_H))



""" # Check if V is orthogonal (Transposed Matrix must be Inverse)
error_orth = torch.linalg.norm(torch.t(V) - V - torch.eye(n))
print("Test if V is orthogonal, Error of norm: ")
print(torch.t(V) - V) """