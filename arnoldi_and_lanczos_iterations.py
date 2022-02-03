import torch


def arnoldi_iteration(A, b, m):
    """Computes a orthogonal basis of the (m+1)-Krylov subspace of A: the space
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


def lanczos_iteration(A, b, m):
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
    # Normalizing input vector b
    V[:, 1] = b/torch.linalg.norm(b, 2)
    for j in range(m):
        # Multiply Matrix A each time with new Vector v to get new candidate vector w
        w = A @ V[:, j]
        # Subtract the projections on previous vectors
        T[j, j] = torch.t(V[:, j]) @ w
        if j+1 < m:
            w = w - T[j+1, j]*V[:, j] - T[j, j]*V[:, j]
            # Normalizing vector w
            T[j+1, j] = torch.linalg.norm(w, 2)
            T[j, j+1] = torch.linalg.norm(w, 2)
            V[:, j+1] = w/T[j+1, j]
        """ 
        #I´m not sure if this if-statement is necessary or even correct

        if T[j, j-1] > eps and T[j-1, j] > eps:  # checking if rounded 0
            V[:, j+1] = w/T[j, j-1]
        else:  # if it happens stop iterating
            return V, T 
        """
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
