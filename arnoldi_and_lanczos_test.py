import pytest
import torch
from arnoldi_and_lanczos_iterations import *


@pytest.fixture
def _A():
    # create matrix A with known eigenvalues 1 to n
    n = 5
    eigvals = torch.linspace(1., n, n)
    eigvecs = torch.randn(n, n)
    A = torch.linalg.solve(eigvecs, (torch.diag(eigvals) @ eigvecs))
    yield A


@pytest.fixture
def _A_symm():
    # create s symmetric matrix A with known eigenvalues 1 to n
    n = 5
    eigvals = torch.linspace(1., n, n)
    eigvecs = torch.randn(n, n)
    # orthogonalize eigenvectors
    eigvecs, R = torch.linalg.qr(eigvecs)
    yield torch.linalg.solve(eigvecs, (torch.diag(eigvals) @ eigvecs))


@pytest.fixture
def _b():
    # create random starting vector b
    n = 5
    b = torch.randn(n)
    yield b/torch.linalg.norm(b)


class TestArnoldiIteration:
    def test_without_dimension_reduction(self, _A, _b):
        A = _A
        b = _b
        m = A.shape[0]
        V, H = arnoldi_iteration(A, b, m)
        # Testing if V^(T)*A*V=H
        test_H = torch.t(V) @ A @ V
        assert torch.allclose(test_H, H, rtol=1e-03, atol=1e-03)

    def test_with_dimension_reduction(self, _A, _b):
        A = _A
        b = _b
        m = 3
        V, H = arnoldi_iteration(A, b, m)
        # Testing if V^(T)*A*V=H
        test_H = torch.t(V) @ A @ V
        assert torch.allclose(test_H, H, rtol=1e-03, atol=1e-03)


class TestLanczosIteration:
    def test_without_dimension_reduction(self, _A_symm, _b):
        A = _A_symm
        b = _b
        m = A.shape[0]
        V, T = lanczos_iteration(A, b, m)
        # Testing if V^(T)*A*V=T
        assert torch.allclose(torch.t(V) @ A @ V, T, rtol=1e-03, atol=1e-03)

    def test_with_dimension_reduction(self, _A_symm, _b):
        A = _A_symm
        b = _b
        m = 3
        V, T = lanczos_iteration(A, b, m)
        # Testing if V^(T)*A*V=T
        assert torch.allclose(torch.t(V) @ A @ V, T, rtol=1e-03, atol=1e-03)
