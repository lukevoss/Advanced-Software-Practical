import torch


def arnoldi_iteration(A, b, m):
    """Arnoldi-Modified Gram-Schmidt Iteration:
    Computes a orthogonal basis of the Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^n b}.

    Arguments
    A:  n × n array
    b:  initial vector (length n)
    m:  dimension of Krylov subspace, must be >= 1

    Returns
    V:  n x m array, the columns are an orthonormal basis of the
        Krylov subspace.
    H:  m x m array, A on basis V. It is upper Hessenberg.  
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
        for i in range(j):  # Subtract the projections on previous vectors
            H[i, j] = torch.t(V[:, i]) @ w
            w = w - H[i, j]*V[:, i]
        # Normalizing vector w
        H[j+1, j] = torch.linalg.norm(w, 2)
        if H[j+1, j] > eps:  # checking if rounded 0
            V[:, j+1] = w/H[j+1, j]
        else:  # if it happens stop iterating
            return V, H
    return V, H
