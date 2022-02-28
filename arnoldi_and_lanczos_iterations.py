from asyncio import base_tasks
import torch

# Arnoldi Iteration


def arnoldi_iteration_gram_schmidt(A, b, m):
    """Arnoldi-Modified Gram-Schmidt

    Computes a orthogonal basis of the (m+1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^m b}.

    Arguments
    A:  n × n torch array
    b:  initial torch vector (length n)
    m:  dimension of Krylov subspace (int), must be >= 1

    Returns
    V:  n x m torch array, the columns are an orthonormal basis of the
        Krylov subspace.
    H:  m x m torch array, A on basis V. It is upper Hessenberg.

    Cost:
    3/2(m^2-m+1)n flops
    """

    eps = 1e-12  # allowed rounding Error
    n = A.shape[0]
    V = torch.zeros(n, m)
    H = torch.zeros(m, m)
    # Normalizing input vector b
    V[:, 0] = b/torch.linalg.norm(b, 2)
    for j in range(m):
        for i in range(j+1):  # Subtract the projections on previous vectors
            H[i, j] = (A @ V[:, j]) @ V[:, i]
        w = A @ V[:, j] - V @ H[:, j]
        if j+1 < m:
            H[j+1, j] = torch.linalg.norm(w, 2)
            if H[j+1, j] > eps:  # checking if rounded 0
                V[:, j+1] = w/H[j+1, j]
            else:  # if it happens stop iterating
                return V, H
    return V, H


def arnoldi_iteration_modified(A, b, m):
    """Arnoldi-Modified Gram-Schmidt

    Computes a orthogonal basis of the (m+1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^m b}.

    Arguments
    A:  n × n torch array
    b:  initial torch vector (length n)
    m:  dimension of Krylov subspace (int), must be >= 1

    Returns
    V:  n x m torch array, the columns are an orthonormal basis of the
        Krylov subspace.
    H:  m x m torch array, A on basis V. It is upper Hessenberg.

    Cost:
    3/2(m^2-m+1)n flops
    """

    eps = 1e-12  # allowed rounding Error
    n = A.shape[0]
    V = torch.zeros(n, m)
    H = torch.zeros(m, m)
    # Normalizing input vector b
    V[:, 0] = b/torch.linalg.norm(b, 2)
    for j in range(m):
        # Multiply Matrix A each time with new Vector v to get new candidate vector w
        w = A @ V[:, j]
        for i in range(j+1):  # Subtract the projections on previous vectors
            H[i, j] = torch.t(V[:, i]) @ w
            w = w - H[i, j]*V[:, i]
        # Normalizing vector w
        if j+1 < m:
            H[j+1, j] = torch.linalg.norm(w, 2)
            if H[j+1, j] > eps:  # checking if rounded 0
                V[:, j+1] = w/H[j+1, j]
            else:  # if it happens stop iterating
                return V, H
    return V, H


def arnoldi_iteration_reorthogonalized(A, b, m):
    """Arnoldi-Modified Gram-Schmidt with reorthogonalistation

    Computes a orthogonal basis of the (m+1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^m b}.

    Arguments
    A:  n × n torch array
    b:  initial torch vector (length n)
    m:  dimension of Krylov subspace (int), must be >= 1

    Returns
    V:  n x m torch array, the columns are an orthonormal basis of the
        Krylov subspace.
    H:  m x m torch array, A on basis V. It is upper Hessenberg.

    Cost:
    3/2(m^2-m+1)n flops
    """

    eps = 1e-12  # allowed rounding Error
    n = A.shape[0]
    V = torch.zeros(n, m)
    H = torch.zeros(m, m)
    # Normalizing input vector b
    V[:, 0] = b/torch.linalg.norm(b, 2)
    for j in range(m):
        # Multiply Matrix A each time with new Vector v to get new candidate vector w
        w = A @ V[:, j]
        initialNorm_w = torch.linalg.norm(w, 2)
        for i in range(j+1):  # Subtract the projections on previous vectors
            H[i, j] = torch.t(V[:, i]) @ w
            w = w - H[i, j]*V[:, i]
        # Normalizing vector w
        norm_w = torch.linalg.norm(w, 2)
        differenceNorm = abs(initialNorm_w-norm_w)
        # Reorthogonalistation:
        if differenceNorm/initialNorm_w < 1/100:
            for i in range(j+1):  # Subtract the projections on previous vectors
                temp = torch.t(V[:, i]) @ w
                w = w - temp*V[:, i]
        if j+1 < m:
            H[j+1, j] = torch.linalg.norm(w, 2)
            if H[j+1, j] > eps:  # checking if rounded 0
                V[:, j+1] = w/H[j+1, j]
            else:  # if it happens stop iterating
                return V, H
    return V, H
# Lanczos Iteration


def lanczos_iteration_saad(A, b, m):
    """Computes a orthogonal basis of the Krylov subspace of a symmetric Matrix A:
    the space spanned by {b, Ab, ..., A^n b}.

    Arguments
    A:  Hermitian matrix, n × n torch array
    b:  initial torch vector v1 (length n)
    m:  dimension of Krylov subspace, must be >= 1 (int)

    Returns
    V:  n x m torch array, the columns are an orthonormal basis of the
        Krylov subspace.
    T:  m x m torch array, A on basis V. It is upper Hessenberg.

    Cost:
    3(2m-1)n flops
    """

    # Test if input matrix is hermitian
    if not torch.allclose(torch.t(A), A, rtol=1e-03, atol=1e-05):
        raise ValueError("The Input matrix is not a hermitian matrix")
    eps = 1e-12  # allowed rounding Error
    n = A.shape[0]
    V = torch.zeros(n, m)  # v0 = 0
    T = torch.zeros(m, m)
    # Normalizing input vector b
    #V[:, 0] = b/torch.linalg.norm(b, 2)
    v = b/torch.linalg.norm(b, 2)
    vo = torch.zeros(n)
    beta = torch.zeros(1)
    V[:, 0] = v
    for j in range(m):
        # Multiply Matrix A each time with new Vector v to get new candidate vector w
        w = A @ torch.t(v) - beta * torch.t(vo)
        # Subtract the projections on previous vectors
        alfa = v @ w
        w = w - alfa * torch.t(v)
        # Normalizing vector w
        beta = torch.linalg.norm(w, 2)
        vo = v
        T[j, j] = alfa  # diagonal entries
        if j+1 < m:  # check if we already reached matrix end
            if abs(beta) > eps:  # check if beta is rounded 0
                T[j, j+1] = beta
                T[j+1, j] = beta
                v = w/beta
                V[:, j+1] = v
            else:  # if it happens stop iterating
                return V, T
    return V, T


def lanczos_iteration_niesen_wright(A, b, m):
    """Computes a orthogonal basis of the Krylov subspace of a symmetric Matrix A:
    the space spanned by {b, Ab, ..., A^n b}.

    Arguments
    A:  Hermitian matrix, n × n array
    b:  initial vector v1 (length n)
    m:  dimension of Krylov subspace, must be >= 1

    Returns
    V:  n x m array, the columns are an orthonormal basis of the
        Krylov subspace.
    T:  m x m array, A on basis V. It is upper Hessenberg.

    Cost:
    3(2m-1)n flops
    """
    if not torch.allclose(torch.t(A), A, rtol=1e-03, atol=1e-05):
        raise ValueError("The Input matrix is not a hermitian matrix")
    eps = 1e-12  # allowed rounding Error
    n = A.shape[0]
    V = torch.zeros(n, m)
    T = torch.zeros(m, m)
    # Normalizing input vector b
    V[:, 0] = b/torch.linalg.norm(b, 2)
    for j in range(m):
        # Multiply Matrix A each time with new Vector v to get new candidate vector w
        w = A @ V[:, j]
        # Subtract the projections on previous vectors
        T[j, j] = torch.t(V[:, j]) @ w
        if j == 0:  # calculate V2 for next iteration
            w = torch.t(w) - T[j, j]*V[:, j]
        else:
            w = torch.t(w) - (T[j-1, j] * V[:, j-1]) - (T[j, j]*V[:, j])
        # Normalizing vector w
        beta = torch.linalg.norm(w, 2)
        if j+1 < m:  # doesn't execute for last iteration
            if abs(beta) > eps:  # checking if rounded 0
                T[j, j+1] = beta
                T[j+1, j] = beta
                V[:, j+1] = w/T[j, j+1]
            else:  # if it happens stop iterating
                return V, T
    return V, T


#################################Appendix#######################################


"""
def arnoldi_iteration(A, b, m):
    ""Computes a orthogonal basis of the (m+1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^m b}.

    Arguments
    A:  n × n array
    b:  initial vector (length n)
    m:  dimension of Krylov subspace, must be >= 1

    Returns
    V:  n x (m+1) array, the columns are an orthonormal basis of the
        Krylov subspace.
    H:  (m+1) x m array, A on basis V. It is upper Hessenberg.

    Cost:
    3/2(m^2-m+1)n flops
    ""
    eps = 1e-12  # allowed rounding Error
    n = A.shape[0]
    V = torch.zeros(n, m+1)
    H = torch.zeros(m+1, m)
    # Normalizing input vector b
    V[:, 0] = b/torch.linalg.norm(b, 2)
    for j in range(1, m+1):
        # Multiply Matrix A each time with new Vector v to get new candidate vector w
        w = A @ V[:, j-1]
        for i in range(j):  # Subtract the projections on previous vectors
            H[i, j-1] = torch.t(V[:, i]) @ w
            w = w - H[i, j-1]*V[:, i]
        # Normalizing vector w
        H[j, j-1] = torch.linalg.norm(w, 2)
        if H[j, j-1] > eps:  # checking if rounded 0
            V[:, j] = w/H[j, j-1]
        else:  # if it happens stop iterating
            return V, H
    return V, H"""


def lanczos_iteration_new(A, b, m):
    """Computes a orthogonal basis of the Krylov subspace of a symmetric Matrix A:
    the space spanned by {b, Ab, ..., A^n b}.

    Arguments
    A:  Hermitian matrix, n × n array
    b:  initial vector v1 (length n)
    m:  dimension of Krylov subspace, must be >= 1

    Returns
    V:  n x m array, the columns are an orthonormal basis of the
        Krylov subspace.
    T:  m x m array, A on basis V. It is upper Hessenberg.

    Cost:
    3(2m-1)n flops
    """
    eps = 1e-12  # allowed rounding Error
    n = A.shape[0]
    V = torch.zeros(n, m)  # v0 = 0
    T = torch.zeros(m, m)
    beta = 0
    # Normalizing input vector b
    v = b/torch.linalg.norm(b, 2)
    V[:, 0] = v
    for j in range(m):
        # Multiply Matrix A each time with new Vector v to get new candidate vector w
        w = A @ V[:, j] - beta * V[:, j]

        # Subtract the projections on previous vectors
        T[j, j] = torch.t(V[:, j]) @ w
        if j == 0:  # calculate V2 for next iteration
            w = torch.t(w) - T[j, j]*V[:, j]
            beta = torch.linalg.norm(w, 2)
            V[:, j+1] = w/beta
        else:
            w = torch.t(w) - T[j, j-1]*V[:, j-1] - T[j, j]*V[:, j]
            # Normalizing vector w
            T[j+1, j] = torch.linalg.norm(w, 2)
            T[j, j+1] = torch.linalg.norm(w, 2)
            if j+1 < m:  # doesn't execute for last iteration
                if T[j+1, j] > eps:  # checking if rounded 0
                    V[:, j+1] = w/T[j+1, j]
                else:  # if it happens stop iterating
                    return V, T
    return V, T
