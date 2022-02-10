from numpy import linspace
import torch
import time
import matplotlib.pyplot as plt
from arnoldi_and_lanczos_iterations import *


def main():
    compare()
    return 0


def compare():
    sizeN = range(1, 101)
    meanErrorsArnoldi = torch.zeros(100)
    timeArnoldi = torch.zeros(100)
    #calcultate Error and Speed for different Sizes of n
    for n in sizeN:
        A = torch.randn(n, n)
        b = torch.randn(n)
        # don't perform dimensionality reduction
        start_time = time.perf_counter()
        V, H = arnoldi_iteration(A, b, n)
        end_time = time.perf_counter()
        timeArnoldi[n-1] = end_time - start_time
        meanErrorsArnoldi[n-1] = calculateError(V, H, A)
    plt.figure(1)
    plt.title("Arnoldi Iteration: Different sizes N of matrix A")
    #Different sizes n of matrix A
    plt.subplot(121)
    plt.plot(sizeN, meanErrorsArnoldi)
    plt.ylabel('Mean Error of Matrix Entries')
    plt.xlabel('Dimension n of nxn Matrix')

    plt.subplot(122)
    plt.plot(sizeN, timeArnoldi)
    plt.ylabel('Execution time in s')
    plt.xlabel('Dimension n of nxn Matrix')

    #Different dimension reductions m
    n = 50
    A = torch.randn(n, n)
    b = torch.randn(n)
    meanErrorsArnoldi = torch.zeros(n)
    timeArnoldi = torch.zeros(n)
    sizeM = range(1, 51)
    for m in sizeM:
        start_time = time.perf_counter()
        V, H = arnoldi_iteration(A, b, m)
        end_time = time.perf_counter()
        timeArnoldi[m-1] = end_time - start_time
        meanErrorsArnoldi[m-1] = calculateError(V, H, A)
    plt.figure(2)
    plt.title("Arnoldi Iteration: Different Dimension reductions m")
    plt.subplot(121)
    plt.plot(sizeM, meanErrorsArnoldi)
    plt.ylabel('Mean Error of Matrix Entries')
    plt.xlabel('Reduced Dimension m of 50x50 Matrix')

    plt.subplot(122)
    plt.plot(sizeM, timeArnoldi)
    plt.ylabel('Execution time in s')
    plt.xlabel('Reduced Dimension m of 50x50 Matrix')
    #plt.show()

    #Test changing norm of A by multipling each value with different skalars
    scalar = range(1, 51)
    meanErrorsArnoldi = torch.zeros(50)
    timeArnoldi = torch.zeros(50)
    norms = torch.zeros(50)
    for i in scalar:
        A = torch.randn(50, 50)*i*2
        b = torch.randn(50)*i*2
        norms[i-1]=torch.norm(A)
        # don't perform dimensionality reduction
        start_time = time.perf_counter()
        V, H = arnoldi_iteration(A, b, 50)
        end_time = time.perf_counter()
        timeArnoldi[i-1] = end_time - start_time
        meanErrorsArnoldi[i-1] = calculateError(V, H, A)

    plt.figure(3)
    plt.title("Arnoldi Iteration: Different Norms of Matrix A")
    plt.subplot(121)
    plt.plot(norms, meanErrorsArnoldi)
    plt.ylabel('Mean Error of Matrix Entries')
    plt.xlabel('Norm of Matrix A')

    plt.subplot(122)
    plt.plot(norms, timeArnoldi)
    plt.ylabel('Execution time in s')
    plt.xlabel('Norm of Matrix A')
    plt.show()


def calculateError(V, H, A):
    """calculates the mean error of entries"""
    n = A.shape[0]
    # Testing if V^(T)*A*V=H
    test_H = torch.t(V) @ A @ V
    errorMatrix = abs(test_H - H)
    meanError = torch.sum(errorMatrix)/(n*n)
    return meanError

# def measureStorage


if __name__ == "__main__":
    main()
