import unittest
import torch
from arnoldi_and_lanczos_iterations import *


class TestArnoldiIteration(unittest.TestCase):
    def setUp(self):
        """ # create Matrix A with known Eigenvalues 1 to n=25
        self.n = 25
        eigvals = torch.linspace(1., n, n)
        eigvecs = torch.randn(n, n)
        self.A = torch.linalg.solve(eigvecs, (torch.diag(eigvals) @ eigvecs))
        # double check if created Matrix A has Eigenvalues 1 to 25
        eigvals, eigvecs = torch.linalg.eig(A)
        # Picking a starting vector and then normalizing it
        b = torch.randn(n)
        self.b = b/torch.linalg.norm(b) """

    def test_without_dimension_reduction(self):
        """ m = self.n
        V, H = arnoldi_iteration(self.A, self.b, m)
        eigvals_aprox, eigvecs_aprox = torch.linalg.eig(H)
        self.eigvals = self.eigvals.sort """


class TestLanczosIteration(unittest.TestCase):
