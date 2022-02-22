from joblib import parallel_backend
import torch
import time
import tracemalloc
import matplotlib.pyplot as plt
from arnoldi_and_lanczos_iterations import *

# TODO: Measure storage consumption
# TODO: Check Iterations for bug in V Matrix, or analyse numerical problem of calculating VTAV=H
# TODO: New Error Measurement
# TODO: Test if algorithms work for non quadratic matrices


def main():
    """
    To compare different Iterative-methods for approximating Eigenvalues and Eigenvectors,
    choose the algorithms to compare and enter them in either of the given arrays.
    Functions must be imported first

    Compared methods need to return the matrices: V, H
    for which the following equation must be valid:
        V^(T)* A * V = H (Saad, Iterative Methods for Sparse Linear Systems, Equation 6.8)

    General Matrix Solvers (e.g Arnoldi-Iteration) represent solvers, 
    that work with general non-hermitian matrices. 
    Hermitian Matrix Solvers (e.g. Lanczos-Iteration) just work with hermitian matrices, 
    meaning when the matrices are symmetric. 
    """
    general_matrix_solvers = [arnoldi_iteration]
    hermitian_matrix_solvers = [arnoldi_iteration, lanczos_iteration_saad]
    # if hermitian_matrix_solvers are being compared: hermitian = True
    compare(hermitian_matrix_solvers, hermitian=True)
    return 0


def compare(solvers, hermitian=False):
    """
    First calcultate error and speed for different sizes of matrix A
    Different sizes of n from 1 to 100
    """
    plt.figure("Compare impact of size of matrix A")
    # number of solvers being compared
    n_comparisons = len(solvers)
    n = 100
    sizeN = range(1, n+1)
    # save mean error and execution time for every compared method
    meanErrors = torch.zeros(n_comparisons, n)
    executionTime = torch.zeros(n_comparisons, n)
    peakStorage = torch.zeros(n_comparisons, n)
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
            tracemalloc.start()  # tracking memory consumption
            V, H = f(A, b, n)  # execute iterative solver method
            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peakStorage[i, n-1] = peak
            executionTime[i, n-1] = end_time - start_time
            meanErrors[i, n-1] = calculateError(H, A)

        # Plot different methods in same subplot
        # Comparing dimensions of matrix A to mean error of matrix H
        plt.subplot(121)
        plt.plot(sizeN, meanErrors[i, :], label=f.__name__)
        plt.ylabel('Mean Error of Matrix Entries')
        plt.xlabel('Dimension n of nxn Matrix')
        plt.legend()
        #plt.title("Mean Error")

        # Comparing dimensions of matrix A to execution time
        plt.subplot(122)
        plt.plot(sizeN, executionTime[i, :], label=f.__name__)
        plt.ylabel('Execution time in s')
        plt.xlabel('Dimension n of nxn Matrix')
        plt.legend()
        # plt.title("Execution Time")

        # plt.subplot(133)
        # plt.plot(sizeN, peakStorage[i, :], label=f.__name__)
        # plt.ylabel('Peak Storage Consumption in Bytes')
        # plt.xlabel('Dimension n of nxn Matrix')
        # plt.legend()
        # plt.title("Storage consumption")

        plt.suptitle(
            "Compare impact of size of matrix A without Dimension reduction")

        i += 1

    """
    Calcultate error and speed for different dimension reductions of matrix A with size 50x50
    Dimension being reduced from 50 to 1
    """
    plt.figure("Compare impact of Dimension Reduction m")
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
            meanErrors[i, m-1] = calculateError(H, A)

        plt.subplot(121)
        plt.plot(sizeM, meanErrors[i, :], label=f.__name__)
        plt.ylabel('Mean Error of Matrix Entries')
        plt.xlabel('Reduced Dimension m of 50x50 Matrix')
        plt.legend()

        plt.subplot(122)
        plt.plot(sizeM, executionTime[i, :], label=f.__name__)
        plt.ylabel('Execution time in s')
        plt.xlabel('Reduced Dimension m of 50x50 Matrix')
        plt.legend()

        plt.suptitle(
            "Compare impact Dimensionality Reduction with 50x50 Matrix")

        i += 1

    """
    Calcultate error and speed for different Frobenius-norms of random matrix A with size 50x50
    The norm is increased by multiplied scalars ranging from 1 to 100 with stepsize 2 to the matrix A.
    """
    plt.figure("Compare impact of norm of matrix A")
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
        A_new = A
        b_new = b
        for j in scalar:
            # increase Norm of matrix A linearly
            A_new = A_new + A
            b_new = b_new + b
            norms[j-1] = torch.norm(A_new)
            # don't perform dimensionality reduction
            start_time = time.perf_counter()
            V, H = f(A_new, b_new, 50)
            end_time = time.perf_counter()
            executionTime[i, j-1] = end_time - start_time
            meanErrors[i, j-1] = calculateError(H, A_new)

        plt.subplot(121)
        plt.plot(norms, meanErrors[i, :], label=f.__name__)
        plt.ylabel('Mean Error of Matrix Entries')
        plt.xlabel('Frobeniusnorm of Matrix A')
        plt.legend()

        plt.subplot(122)
        plt.plot(norms, executionTime[i, :], label=f.__name__)
        plt.ylabel('Execution time in s')
        plt.xlabel('Frobeniusnorm of Matrix A')
        plt.legend()
        i += 1

        plt.suptitle("Compare impact of Norm of Matrix A (50x50 Matrix)")

    plt.show()


def calculateError(H, A):
    """calculates the mean error of each eigenvalue from Matrix H"""
    # calculate eigenvalues of Matrix A
    eigvals = torch.linalg.eigvals(A)
    # calculate eigenvalues of Matrix H
    eigvals_aprox = torch.linalg.eigvals(H)
    nEigvals = len(eigvals_aprox)
    errors = torch.zeros(nEigvals)
    # for each eigenvalue of H find closest eigenvalue of Matrix A
    for i in range(nEigvals):
        error = min(abs(eigvals-eigvals_aprox[i]))
        errors[i] = error
    # calculate mean of errors
    meanError = torch.sum(errors)/(nEigvals)
    return meanError


if __name__ == "__main__":
    main()
