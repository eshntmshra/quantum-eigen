import numpy as np
from scipy.linalg import eigh
import argparse

def build_2d_hamiltonian(N=20, potential='well'):
    """
    Build a discretized 2D Hamiltonian on an N x N grid.
    Parameters
    ----------
    N : int
        Number of points in each dimension (N^2 total points).
    potential : str
        Choose the potential. 'well' or 'harmonic' examples.
    Returns
    -------
    H : ndarray of shape (N^2, N^2)
        The Hamiltonian matrix approximating -d^2/dx^2 - d^2/dy^2 + V(x,y).
    """
    dx = 1. / float(N)
    inv_dx2 = float(N * N)
    H = np.zeros((N*N, N*N), dtype=np.float64)

    def idx(i, j):
        return i * N + j

    def V(i, j):
        if potential == 'well':
            return 0.
        elif potential == 'harmonic':
            x = (i - N/2) * dx
            y = (j - N/2) * dx
            return 4. * (x**2 + y**2)
        else:
            return 0.

    for i in range(N):
        for j in range(N):
            row = idx(i, j)
            H[row, row] = -4. * inv_dx2 + V(i, j)
            if i > 0:
                H[row, idx(i-1, j)] = inv_dx2
            if i < N-1:
                H[row, idx(i+1, j)] = inv_dx2
            if j > 0:
                H[row, idx(i, j-1)] = inv_dx2
            if j < N-1:
                H[row, idx(i, j+1)] = inv_dx2
    return H


def solve_eigen(N=20, potential='well', n_eigs=None):
    """
    Build a 2D Hamiltonian and solve for the lowest n_eigs eigenvalues.
    Parameters
    ----------
    N : int
        Grid points in each dimension.
    potential : str
        Potential type.
    n_eigs : int
        Number of eigenvalues to return.
    Returns
    -------
    vals : array_like
        The lowest n_eigs eigenvalues sorted ascending.
    vecs : array_like
        The corresponding eigenvectors.
    """
    # Sanity checks
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer")
    if potential not in ('well', 'harmonic'):
        raise ValueError("potential must be 'well' or 'harmonic'")
    if n_eigs is not None:
        if not isinstance(n_eigs, int) or n_eigs <= 0:
            raise ValueError("n_eigs must be a positive integer")
        if n_eigs > N**2:
            raise ValueError(f"n_eigs ({n_eigs}) cannot exceed N^2 ({N**2})")

    H = build_2d_hamiltonian(N, potential)
    vals, vecs = eigh(H)

    idx_sorted = np.argsort(vals)
    vals_sorted = vals[idx_sorted]
    vecs_sorted = vecs[:, idx_sorted]

    if n_eigs is None:
        return vals_sorted, vecs_sorted
    else:
        return vals_sorted[:n_eigs], vecs_sorted[:, :n_eigs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D Hamiltonian Eigenvalue Solver')
    parser.add_argument('--N',         type=int,   default=10,       help='Grid points per dimension (default: 10)')
    parser.add_argument('--potential', type=str,   default='well',   choices=['well', 'harmonic'], help='Potential type (default: well)')
    parser.add_argument('--n-eigs',    type=int,   default=5,        help='Number of eigenvalues to return (default: 5)')
    parser.add_argument('--out',       type=str,   default=None,     help='Output file to save eigenvalues')
    args = parser.parse_args()

    vals, vecs = solve_eigen(N=args.N, potential=args.potential, n_eigs=args.n_eigs)
    print(f"Lowest {args.n_eigs} eigenvalues:", vals)

    if args.out:
        np.savetxt(args.out, vals)
        print(f"Saved to {args.out}")