import torch


############################# Arnoldi Iterations #################################

def arnoldi_iteration_gram_schmidt(A, b, m):
    """Arnoldi-Gram-Schmidt Iteration

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
        Flops: 2*n*m^2
        Storage: (m + 1)n

    Based on:
    Iterative Methods for Sparse Linear Systems, Yousef Saad, Algorithm 6.1
    """

    eps = 1e-10  # allowed rounding Error
    n = A.shape[0]
    V = torch.zeros(n, m)
    H = torch.zeros(m, m)
    # Normalizing input vector b
    V[:, 0] = b/torch.linalg.norm(b, 2)

    for j in range(m):
        # Gram-Schmidt orthogonalization for each new created Vector w (H[i,j])
        for i in range(j+1):
            H[i, j] = (A @ V[:, j]) @ V[:, i]
        w = A @ V[:, j] - V @ H[:, j]
        # Inserting the rest of calculated values in matrices
        if j+1 < m:
            H[j+1, j] = torch.linalg.norm(w, 2)
            if H[j+1, j] > eps:  # checking if rounded 0
                V[:, j+1] = w/H[j+1, j]  # normalizing w
            else:  # if rounded 0 stop iterating
                return V, H
    return V, H


def arnoldi_iteration_modified(A, b, m):
    """Arnoldi-Modified Gram-Schmidt Iteration

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
        Flops: 2*n*m^2
        Storage: (m + 1)n

    Based on:
    Iterative Methods for Sparse Linear Systems, Yousef Saad, 
    Algorithm 6.2
    and
    Algorithm 919: A Krylov Subspace Algorithm for Evaluating 
    the ϕ-Functions Appearing in Exponential Integrators, 
    Jitse Niesen and Will M. Wright, Algorithm 1
    """

    eps = 1e-10  # allowed rounding Error
    n = A.shape[0]
    V = torch.zeros(n, m)
    H = torch.zeros(m, m)
    # Normalizing input vector b
    V[:, 0] = b/torch.linalg.norm(b, 2)
    for j in range(m):
        # Multiply Matrix A each time with new Vector v to get new candidate vector w
        w = A @ V[:, j]
        # Modified Gram-Schmidt orthogonalization of new vector w
        for i in range(j+1):
            H[i, j] = torch.t(V[:, i]) @ w
            w = w - H[i, j]*V[:, i]
        # Insert the rest of calculated values in matrices
        if j+1 < m:
            # Normalizing vector w
            H[j+1, j] = torch.linalg.norm(w, 2)
            if H[j+1, j] > eps:  # checking if rounded 0
                V[:, j+1] = w/H[j+1, j]  # normalizing w
            else:  # if rounded 0 stop iterating
                return V, H
    return V, H


def arnoldi_iteration_reorthogonalized(A, b, m):
    """Arnoldi-Modified Gram-Schmidt Iteration with reorthogonalistation 

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
        Flops: 4*n*m^2
        Storage: (m + 1)n

    Based on:
    Iterative Methods for Sparse Linear Systems, Yousef Saad, 
    6.3.2 Practical Implementations
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
        # Modified Gram-Schmidt orthogonalization of new vector w
        for i in range(j+1):
            H[i, j] = torch.t(V[:, i]) @ w
            w = w - H[i, j]*V[:, i]
        # compare initial and new Norm of w
        norm_w = torch.linalg.norm(w, 2)
        differenceNorm = abs(initialNorm_w-norm_w)
        # Reorthogonalistation if difference of Norms is small (Possible severe cancelations)
        if differenceNorm/initialNorm_w < 1/100:
            for i in range(j+1):  # Subtract the projections on previous vectors
                temp = torch.t(V[:, i]) @ w
                w = w - temp*V[:, i]
        # Insert the rest of calculated values in matrices
        if j+1 < m:
            H[j+1, j] = torch.linalg.norm(w, 2)
            if H[j+1, j] > eps:  # checking if rounded 0
                V[:, j+1] = w/H[j+1, j]  # normalizing w
            else:  # if it happens stop iterating
                return V, H
    return V, H


############################# Lanczos Iterations #################################


def lanczos_iteration_saad(A, b, m):
    """Computes a orthogonal basis of the Krylov subspace of a hermitian Matrix A:
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

    Based on:
    Iterative Methods for Sparse Linear Systems, Yousef Saad, 
    Algorithm 6.6
    """

    # Test if input matrix is hermitian
    if not torch.allclose(torch.t(A), A, rtol=1e-03, atol=1e-05):
        raise ValueError("The Input matrix is not a hermitian matrix")
    eps = 1e-12  # allowed rounding Error
    n = A.shape[0]
    V = torch.zeros(n, m)  # v0 = 0
    T = torch.zeros(m, m)
    # Normalizing input vector b
    v = b/torch.linalg.norm(b, 2)
    vo = torch.zeros(n)
    beta = torch.zeros(1)
    V[:, 0] = v
    for j in range(m):
        # Multiply Matrix A each time with new Vector v to get new candidate vector w
        # and orthogonalize new vector w 
        w = A @ torch.t(v) - beta * torch.t(vo)
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
            else:  # if rounded 0 stop iterating
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

    Based on:
    Algorithm 919: A Krylov Subspace Algorithm for Evaluating 
    the ϕ-Functions Appearing in Exponential Integrators, 
    Jitse Niesen and Will M. Wright, Algorithm 2
    """
    # Test if input matrix is hermitian
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
        # orthogonalization of new vector w
        T[j, j] = torch.t(V[:, j]) @ w
        if j == 0:  # skip (T[j-1, j] * V[:, j-1]) for first iteration
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
            else:  # if rounded 0 stop iterating
                return V, T
    return V, T


