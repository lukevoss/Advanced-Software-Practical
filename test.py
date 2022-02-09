import torch
from arnoldi_and_lanczos_iterations import *

# First testing of Arnoldi Iteration:

# create matrix A with known eigenvalues 1 to n=25
n = 5
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
m = n
V, H = arnoldi_iteration(A, b, m)
eigvals_aprox, eigvecs_aprox = torch.linalg.eig(H)
print("Approximated Eigenvalues with no Dimension Reduction:")
print(eigvals_aprox)
test_H = torch.t(V) @ A @ V
error = abs(test_H - H)
print(error)
print(torch.isclose(torch.t(V) @ A @ V, H, rtol= 1e-03, atol=1e-05))


# Testing Arnoldi algorithm with m=10
m = 5
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

# Testing of Lanczos iteration:

# To work with symmetric matrix, orthogonalize eigvecs
eigvecs, R = torch.linalg.qr(eigvecs)
A = torch.linalg.solve(eigvecs, (torch.diag(eigvals) @ eigvecs))
print("created symmetric matrix A:")
print(A)
# double check if created Matrix A has Eigenvalues 1 to 25
eigvals_test, eigvecs_test = torch.linalg.eig(A)
print("Eigenvalues of created symmetric Matrix:")
print(eigvals_test)

# Testing Lanczos algorithm with no dimensionality reduction
m = n
V, T = lanczos_iteration(A, b, m)
#print("created matrix T:")
# print(T)
eigvals_aprox, eigvecs_aprox = torch.linalg.eig(T)
print("Approximated Eigenvalues with no Dimension Reduction:")
print(eigvals_aprox)


# Testing if V^(T)*A*V=T
norm_VAV = torch.linalg.norm(torch.t(V) @ A @ V)
norm_H = torch.linalg.norm(T)
print("Test of two norms")
print(norm_VAV)
print(norm_H)
print("Error of norm:")
print(abs(norm_VAV-norm_H))

# Testing Lanczos algorithm with dimensionality reduction
m =5
V, T = lanczos_iteration(A, b, m)
eigvals_aprox, eigvecs_aprox = torch.linalg.eig(T)
print("Approximated Eigenvalues with no Dimension Reduction:")
print(eigvals_aprox)
test_T = torch.t(V) @ A @ V
error = abs(test_T - T)
print("Error")
print(error)
print("VAV")
print(test_T)
print(T)
print(torch.isclose(torch.t(V) @ A @ V, T,  rtol=1e-03, atol=1e-05))
