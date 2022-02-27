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
    general_matrix_solvers = [arnoldi_iteration_gram_schmidt, arnoldi_iteration_modified]
    hermitian_matrix_solvers = [lanczos_iteration_niesen_wright, lanczos_iteration_saad]
    # if hermitian_matrix_solvers are being compared: hermitian = True
    compare(general_matrix_solvers, hermitian=False)
    return 0


def compare(solvers, hermitian=False):
    evalutationTimes = 10
    """
    First calcultate error and speed for different sizes of matrix A
    Different sizes of n from 1 to 100
    """
    plt.figure("Compare impact of size of matrix A")
    # number of solvers being compared
    n_comparisons = len(solvers)
    n = 100
    stepsize = 5

    sizeN = range(5, n+1, stepsize)
    # save mean error and execution time for every compared method
    errors = torch.zeros(n_comparisons, len(sizeN))
    executionTime = torch.zeros(n_comparisons, len(sizeN))
    # peakStorage = torch.zeros(n_comparisons, len(sizeN))
    standardDeviationError = torch.zeros(n_comparisons, len(sizeN))
    standardDeviationTime = torch.zeros(n_comparisons, len(sizeN))
    i = 0
    # iterate over methods
    for f in solvers:
        for n in sizeN:  # iterate from 1x1 to nxn size Matrix A
            meanError = torch.zeros(evalutationTimes)
            meanExecutionTime = torch.zeros(evalutationTimes)
            # meanPeakStorage = torch.zeros(evalutationTimes)
            for j in range(evalutationTimes):
                # if we compare hermitian methods, create random hermitian Matrix
                if hermitian == True:
                    A = torch.randn(n, n)
                    A = A + torch.t(A)
                else:
                    A = torch.randn(n, n)
                b = torch.randn(n)  # starting vector b
                # don't perform dimensionality reduction to see impact of size of Matrix A
                start_time = time.perf_counter()  # measure execution time
                # tracemalloc.start()  # tracking memory consumption
                V, H = f(A, b, n)  # execute iterative solver method
                end_time = time.perf_counter()
                # current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                # meanPeakStorage[j] = peak
                meanExecutionTime[j] = end_time - start_time
                meanError[j] = calculateError(H, A)
            # peakStorage[i, (n //
                # stepsize)-1] = torch.sum(meanPeakStorage)/evalutationTimes
            index = (n//stepsize) - 1
            executionTime[i, index] = torch.sum(
                meanExecutionTime)/evalutationTimes
            errors[i, index] = torch.sum(meanError)/evalutationTimes
            standardDeviationError[i, index] = torch.std(
                meanError, unbiased=False)
            standardDeviationTime[i, index] = torch.std(
                meanExecutionTime, unbiased=False)

        # Plot different methods in same subplot
        # Comparing dimensions of matrix A to mean error of matrix H
        errors_y1 = errors[i, :] + standardDeviationError[i, :]
        errors_y2 = errors[i, :] - standardDeviationError[i, :]
        plt.subplot(121)
        plt.fill_between(sizeN, errors_y1, errors_y2, alpha=.5)
        plt.plot(sizeN, errors[i, :], label=f.__name__)
        plt.ylabel('Mean Error of Matrix Entries')
        plt.xlabel('Dimension n of nxn Matrix')
        plt.legend()
        #plt.title("Mean Error")

        # Comparing dimensions of matrix A to execution time
        time_y1 = executionTime[i, :] + standardDeviationTime[i, :]
        time_y2 = executionTime[i, :] - standardDeviationTime[i, :]
        plt.subplot(122)
        plt.fill_between(sizeN, time_y1, time_y2, alpha=.5)
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
    stepsize = 5
    sizeM = range(stepsize, n+1, stepsize)
    errors = torch.zeros(n_comparisons, len(sizeM))
    executionTime = torch.zeros(n_comparisons, len(sizeM))
    standardDeviationError = torch.zeros(n_comparisons, len(sizeM))
    standardDeviationTime = torch.zeros(n_comparisons, len(sizeM))
    i = 0
    for f in solvers:
        for m in sizeM:
            meanError = torch.zeros(evalutationTimes)
            meanExecutionTime = torch.zeros(evalutationTimes)
            # meanPeakStorage = torch.zeros(evalutationTimes)
            for j in range(evalutationTimes):
                if hermitian == True:
                    A = torch.randn(n, n)
                    A = A + torch.t(A)
                else:
                    A = torch.randn(n, n)
                b = torch.randn(n)
                start_time = time.perf_counter()
                V, H = f(A, b, m)
                end_time = time.perf_counter()
                meanExecutionTime[j] = end_time - start_time
                meanError[j] = calculateError(H, A)
            index = (m//stepsize)-1
            executionTime[i, index] = torch.sum(
                meanExecutionTime)/evalutationTimes
            errors[i, index] = torch.sum(meanError)/evalutationTimes
            standardDeviationError[i, index] = torch.std(
                meanError, unbiased=False)
            standardDeviationTime[i, index] = torch.std(
                meanExecutionTime, unbiased=False)

        errors_y1 = errors[i, :] + standardDeviationError[i, :]
        errors_y2 = errors[i, :] - standardDeviationError[i, :]
        plt.subplot(121)
        plt.fill_between(sizeM, errors_y1, errors_y2, alpha=.5)
        plt.plot(sizeM, errors[i, :], label=f.__name__)
        plt.ylabel('Mean Error of Matrix Entries')
        plt.xlabel('Reduced Dimension m of 50x50 Matrix')
        plt.legend()

        time_y1 = executionTime[i, :] + standardDeviationTime[i, :]
        time_y2 = executionTime[i, :] - standardDeviationTime[i, :]
        plt.subplot(122)
        plt.fill_between(sizeM, time_y1, time_y2, alpha=.5)
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
    stepsize = 10
    scalarMax = 300
    scalar = range(stepsize, scalarMax+1, stepsize)
    errors = torch.zeros(n_comparisons, len(scalar))
    executionTime = torch.zeros(n_comparisons, len(scalar))
    norms = torch.zeros(len(scalar))
    standardDeviationError = torch.zeros(n_comparisons, len(scalar))
    standardDeviationTime = torch.zeros(n_comparisons, len(scalar))
    i = 0
    for f in solvers:
        for s in scalar:
            meanError = torch.zeros(evalutationTimes)
            meanExecutionTime = torch.zeros(evalutationTimes)
            meanNorm = torch.zeros(evalutationTimes)
            # meanPeakStorage = torch.zeros(evalutationTimes)
            for j in range(evalutationTimes):
                if hermitian == True:
                    A = torch.randn(n, n)*s
                    A = A + torch.t(A)
                else:
                    A = torch.randn(n, n)*s
                b = torch.randn(n)*s
                # increase Norm of matrix A linearly
                meanNorm[j] = torch.norm(A)
                # don't perform dimensionality reduction
                start_time = time.perf_counter()
                V, H = f(A, b, 50)
                end_time = time.perf_counter()
                meanExecutionTime[j] = end_time - start_time
                meanError[j] = calculateError(H, A)
            index = (s//stepsize)-1
            norms[index] = torch.sum(meanNorm)/evalutationTimes
            executionTime[i, index] = torch.sum(
                meanExecutionTime)/evalutationTimes
            errors[i, index] = torch.sum(meanError)/evalutationTimes
            standardDeviationError[i, index] = torch.std(
                meanError, unbiased=False)
            standardDeviationTime[i, index] = torch.std(
                meanExecutionTime, unbiased=False)

        errors_y1 = errors[i, :] + standardDeviationError[i, :]
        errors_y2 = errors[i, :] - standardDeviationError[i, :]
        plt.subplot(121)
        plt.fill_between(norms, errors_y1, errors_y2, alpha=.5)
        plt.plot(norms, errors[i, :], label=f.__name__)
        plt.ylabel('Mean Error of Matrix Entries')
        plt.xlabel('Frobeniusnorm of Matrix A')
        plt.legend()

        time_y1 = executionTime[i, :] + standardDeviationTime[i, :]
        time_y2 = executionTime[i, :] - standardDeviationTime[i, :]
        plt.subplot(122)
        plt.fill_between(norms, time_y1, time_y2, alpha=.5)
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
