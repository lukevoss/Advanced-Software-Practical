import pytest
import torch
from arnoldi_and_lanczos_iterations import *

"""
Testing file for Algorithms in arnoldi_and_lanczos_iterations.py


For large matrices the problem arises that the calculation V^T*A*V is numerically inaccurate. 
This is especially the case for values which are equal to 0 in the H or T matrix.

    Arnoldi-Iteration:
        H = 
        low error ->    [-2, -1,  1, 7]<-low error
                        [ 7,  3, -3, 2]
                        [ 0,  3,  5, 4]
        higher error->  [ 0,  0,  3, 6]

    Lanczos-Iteration
        T = 
        low error ->    [-2, -1,  0, 0]<-higher error
                        [ 7,  3, -3, 0]
                        [ 0,  3,  5, 4]
        higher error->  [ 0,  0,  3, 6]
"""


@pytest.fixture
def _A():
    # create a random nxn matrix A
    n = 5
    yield torch.randn(n, n)


@pytest.fixture
def _A_symm():
    # create a random symmetric nxn matrix A
    n = 5
    A = torch.randn(n, n)
    yield A + torch.t(A)


@pytest.fixture
def _b():
    # create random starting vector b
    n = 5
    b = torch.randn(n)
    yield b/torch.linalg.norm(b)


class TestArnoldiIterationGramSchmidt:
    def test_without_dimension_reduction(self, _A, _b):
        """
        Test Arnoldi iteration without dimension reduction

        Test by Equation:
        V^(T)* A * V = H (Saad, Iterative Methods for Sparse Linear Systems, Equation 6.8)
        """
        A = _A
        b = _b
        m = A.shape[0]
        V, H = arnoldi_iteration_gram_schmidt(A, b, m)
        # Testing if V^(T)*A*V=H
        test_H = torch.t(V) @ A @ V
        # check if entries of matrices are close
        assert torch.allclose(test_H, H, rtol=1e-03, atol=1e-03)

    def test_with_dimension_reduction(self, _A, _b):
        """
        Test Arnoldi iteration with dimension reduction

        Test by Equation 
        V^(T)* A * V = H (Saad, Iterative Methods for Sparse Linear Systems, Equation 6.8)
        """
        A = _A
        b = _b
        m = 3
        V, H = arnoldi_iteration_gram_schmidt(A, b, m)
        # Testing if V^(T)*A*V=H
        test_H = torch.t(V) @ A @ V
        # check if entries of matrices are close
        assert torch.allclose(test_H, H, rtol=1e-03, atol=1e-03)


class TestArnoldiIterationModified:
    def test_without_dimension_reduction(self, _A, _b):
        """
        Test Modified Arnoldi iteration without dimension reduction

        Test by Equation:
        V^(T)* A * V = H (Saad, Iterative Methods for Sparse Linear Systems, Equation 6.8)
        """
        A = _A
        b = _b
        m = A.shape[0]
        V, H = arnoldi_iteration_modified(A, b, m)
        # Testing if V^(T)*A*V=H
        test_H = torch.t(V) @ A @ V
        # check if entries of matrices are close
        assert torch.allclose(test_H, H, rtol=1e-03, atol=1e-03)

    def test_with_dimension_reduction(self, _A, _b):
        """
        Test Modified Arnoldi iteration with dimension reduction

        Test by Equation 
        V^(T)* A * V = H (Saad, Iterative Methods for Sparse Linear Systems, Equation 6.8)
        """
        A = _A
        b = _b
        m = 3
        V, H = arnoldi_iteration_modified(A, b, m)
        # Testing if V^(T)*A*V=H
        test_H = torch.t(V) @ A @ V
        # check if entries of matrices are close
        assert torch.allclose(test_H, H, rtol=1e-03, atol=1e-03)


class TestArnoldiIterationGramReorthogonalized:
    def test_without_dimension_reduction(self, _A, _b):
        """
        Test Modified Arnoldi iteration with reorthogonalized without dimension reduction

        Test by Equation:
        V^(T)* A * V = H (Saad, Iterative Methods for Sparse Linear Systems, Equation 6.8)
        """
        A = _A
        b = _b
        m = A.shape[0]
        V, H = arnoldi_iteration_reorthogonalized(A, b, m)
        # Testing if V^(T)*A*V=H
        test_H = torch.t(V) @ A @ V
        # check if entries of matrices are close
        assert torch.allclose(test_H, H, rtol=1e-03, atol=1e-03)

    def test_with_dimension_reduction(self, _A, _b):
        """
        Test Modified Arnoldi iteration with reorthogonalized with dimension reduction

        Test by Equation 
        V^(T)* A * V = H (Saad, Iterative Methods for Sparse Linear Systems, Equation 6.8)
        """
        A = _A
        b = _b
        m = 3
        V, H = arnoldi_iteration_reorthogonalized(A, b, m)
        # Testing if V^(T)*A*V=H
        test_H = torch.t(V) @ A @ V
        # check if entries of matrices are close
        assert torch.allclose(test_H, H, rtol=1e-03, atol=1e-03)


class TestLanczosIterationSaad:
    def test_without_dimension_reduction(self, _A_symm, _b):
        """
        Test Lanczos iteration without dimension reduction

        Test by Equation 
        V^(T)* A * V = H (Saad, Iterative Methods for Sparse Linear Systems, Equation 6.8)
        """
        A = _A_symm
        b = _b
        m = A.shape[0]
        V, T = lanczos_iteration_saad(A, b, m)
        # Testing if V^(T)*A*V=T
        assert torch.allclose(torch.t(V) @ A @ V, T, rtol=1e-03, atol=1e-03)

    def test_with_dimension_reduction(self, _A_symm, _b):
        """
        Test Lanczos iteration with dimension reduction

        Test by Equation 
        V^(T)* A * V = H (Saad, Iterative Methods for Sparse Linear Systems, Equation 6.8)
        """
        A = _A_symm
        b = _b
        m = 3
        V, T = lanczos_iteration_saad(A, b, m)
        # Testing if V^(T)*A*V=T
        assert torch.allclose(torch.t(V) @ A @ V, T, rtol=1e-03, atol=1e-05)

    def test_throw_error(self, _A, _b):
        """
        Test if an error is thrown if not a hermitian matrix is given as input
        """
        A = _A
        b = _b
        m = 3
        # test raising Error
        with pytest.raises(ValueError):
            V, T = lanczos_iteration_saad(A, b, m)


class TestLanczosIterationNiesenWright:
    def test_without_dimension_reduction(self, _A_symm, _b):
        """
        Test Lanczos iteration without dimension reduction

        Test by Equation 
        V^(T)* A * V = H (Saad, Iterative Methods for Sparse Linear Systems, Equation 6.8)
        """
        A = _A_symm
        b = _b
        m = A.shape[0]
        V, T = lanczos_iteration_niesen_wright(A, b, m)
        # Testing if V^(T)*A*V=T
        assert torch.allclose(torch.t(V) @ A @ V, T, rtol=1e-03, atol=1e-03)

    def test_with_dimension_reduction(self, _A_symm, _b):
        """
        Test Lanczos iteration with dimension reduction

        Test by Equation 
        V^(T)* A * V = H (Saad, Iterative Methods for Sparse Linear Systems, Equation 6.8)
        """
        A = _A_symm
        b = _b
        m = 3
        V, T = lanczos_iteration_niesen_wright(A, b, m)
        # Testing if V^(T)*A*V=T
        assert torch.allclose(torch.t(V) @ A @ V, T, rtol=1e-03, atol=1e-05)

    def test_throw_error(self, _A, _b):
        """
        Test if an error is thrown if not a hermitian matrix is given as input
        """
        A = _A
        b = _b
        m = 3
        # test raising Error
        with pytest.raises(ValueError):
            V, T = lanczos_iteration_niesen_wright(A, b, m)
