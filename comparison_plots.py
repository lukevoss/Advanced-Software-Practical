import torch
import time
import matplotlib.pyplot as plt
from arnoldi_and_lanczos_iterations import *

# TODO: Masure storage consumption
# TODO: Check figure 3 for bug
# TODO: Check Iterations for bug in V Matrix, or analyse numerical problem


def main():
    """
    To compare different Iterative-methods for approximating Eigenvalues and Eigenvectors,
    choose the algorithms to compare and enter them in either of the given arrays.
    Functions must be imported first

    General Matrix Solvers (e.g Arnoldi-Iteration) represent solvers, 
    that work with general non-hermitian matrices. 
    Hermitian Matrix Solvers (e.g. Lanczos-Iteration) just work with hermitian matrices, 
    meaning when the matrices are symmetric. 
    """
    general_matrix_solvers = [arnoldi_iteration]
    hermitian_matrix_solvers = [arnoldi_iteration, lanczos_iteration]
    # if hermitian_matrix_solvers are being compared: hermitian = True
    compare(hermitian_matrix_solvers, hermitian=True)
    return 0


def compare(solvers, hermitian=False):
    """
    First calcultate error and speed for different sizes of matrix A
    Different sizes of n from 1 to 100
    """
    plt.figure(1)
    # number of solvers being compared
    n_comparisons = len(solvers)
    n = 100
    sizeN = range(1, n+1)
    # save mean error and execution time for every compared method
    meanErrors = torch.zeros(n_comparisons, n)
    executionTime = torch.zeros(n_comparisons, n)
    i = 0
    # iterate over methods
    for f in solvers:
        for n in sizeN:  # iterate from 1x1 to nxn size Matrix A
            # if we compare hermitian methods, create random hermitian Matrix
            if hermitian == True:
                A = torch.randn(n, n)
                A = A + torch.t(A)
            else:
                A = torch.randn(n, n)
            b = torch.randn(n)  # starting vector b
            # don't perform dimensionality reduction to see impact of size of Matrix A
            start_time = time.perf_counter()  # measure execution time
            V, H = f(A, b, n)  # execute iterative solver method
            end_time = time.perf_counter()
            executionTime[i, n-1] = end_time - start_time
            meanErrors[i, n-1] = calculateError(V, H, A)

        # Plot different methods in same subplot
        # Comparing dimensions of matrix A to mean error of matrix H
        plt.subplot(121)
        plt.plot(sizeN, meanErrors[i, :])
        plt.ylabel('Mean Error of Matrix Entries')
        plt.xlabel('Dimension n of nxn Matrix')
        # Comparing dimensions of matrix A to execution time
        plt.subplot(122)
        plt.plot(sizeN, executionTime[i, :])
        plt.ylabel('Execution time in s')
        plt.xlabel('Dimension n of nxn Matrix')

        i += 1

    """
    Calcultate error and speed for different dimension reductions of matrix A with size 50x50
    Dimension being reduced from 50 to 1
    """
    plt.figure(2)
    n = 50
    if hermitian == True:
        A = torch.randn(n, n)
        A = A + torch.t(A)
    else:
        A = torch.randn(n, n)
    b = torch.randn(n)
    meanErrors = torch.zeros(n_comparisons, n)
    executionTime = torch.zeros(n_comparisons, n)
    sizeM = range(1, 51)
    i = 0
    for f in solvers:
        for m in sizeM:
            start_time = time.perf_counter()
            V, H = f(A, b, m)
            end_time = time.perf_counter()
            executionTime[i, m-1] = end_time - start_time
            meanErrors[i, m-1] = calculateError(V, H, A)

        plt.subplot(121)
        plt.plot(sizeM, meanErrors[i, :])
        plt.ylabel('Mean Error of Matrix Entries')
        plt.xlabel('Reduced Dimension m of 50x50 Matrix')

        plt.subplot(122)
        plt.plot(sizeM, executionTime[i, :])
        plt.ylabel('Execution time in s')
        plt.xlabel('Reduced Dimension m of 50x50 Matrix')

        i += 1

    """
    Calcultate error and speed for different Frobenius-norms of random matrix A with size 50x50
    The norm is increased by multiplied scalars ranging from 1 to 100 with stepsize 2 to the matrix A.
    """
    plt.figure(3)
    scalar = range(1, n+1)
    meanErrors = torch.zeros(n_comparisons, n)
    executionTime = torch.zeros(n_comparisons, n)
    norms = torch.zeros(n)
    if hermitian == True:
        A = torch.randn(n, n)
        A = A + torch.t(A)
    else:
        A = torch.randn(n, n)
    b = torch.randn(n)
    i = 0
    for f in solvers:
        for j in scalar:
            # increase Norm of matrix A
            A = A*j*2
            b = b*j*2
            norms[j-1] = torch.norm(A)
            # don't perform dimensionality reduction
            start_time = time.perf_counter()
            V, H = f(A, b, 50)
            end_time = time.perf_counter()
            executionTime[i, j-1] = end_time - start_time
            meanErrors[i, j-1] = calculateError(V, H, A)

        plt.subplot(121)
        plt.plot(norms, meanErrors[i, :])
        plt.ylabel('Mean Error of Matrix Entries')
        plt.xlabel('Norm of Matrix A')

        plt.subplot(122)
        plt.plot(norms, executionTime[i, :])
        plt.ylabel('Execution time in s')
        plt.xlabel('Norm of Matrix A')

        i += 1
    plt.show()


def calculateError(V, H, A):
    """calculates the mean error of entries"""
    n = A.shape[0]
    # Testing if V^(T)*A*V=H
    test_H = torch.t(V) @ A @ V
    errorMatrix = abs(test_H - H)
    meanError = torch.sum(errorMatrix)/(n*n)
    return meanError


if __name__ == "__main__":
    main()
