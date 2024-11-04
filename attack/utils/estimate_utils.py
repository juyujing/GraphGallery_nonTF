"""The functions in this file are mainly used for attacks"""

import numpy as np
import torch
import scipy.sparse as sp
from numba import njit


def estimate_loss_with_perturbation_gradient(candidates,
                                             adj_matrix,
                                             window_size,
                                             dim,
                                             num_neg=1):
    """Computes the estimated loss using the gradient defined with eigenvalue perturbation.

    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param adj_matrix: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param window_size: int
        Size of the window
    :param dim: int
        Size of the embedding
    :param num_neg: int
        Number of negative samples
    :return:
    """
    adj_matrix = torch.tensor(adj_matrix.toarray(), dtype=torch.float32)

    # Enable gradient tracking
    adj_matrix.requires_grad = True

    deg = torch.sum(adj_matrix, dim=1)
    volume = torch.sum(adj_matrix)

    # Transition matrix
    transition_matrix = adj_matrix / deg[:, None]

    # Compute the power series sum
    sum_of_powers = transition_matrix.clone()
    last = transition_matrix.clone()
    for i in range(1, window_size):
        last = torch.matmul(last, transition_matrix)
        sum_of_powers += last

    M = sum_of_powers / deg[:, None] * volume / (num_neg * window_size)
    logM = torch.log(torch.clamp(M, min=1.0))

    # Frobenius norm of logM
    norm_logM = torch.norm(logM, p='fro') ** 2

    # Eigenvalue decomposition using scipy for sparse matrix
    sp_logM = sp.csr_matrix(logM.detach().numpy())
    _, eigenvecs = sp.linalg.eigsh(sp_logM, dim)

    # Convert eigenvectors to tensor
    eigenvecs = torch.tensor(eigenvecs, dtype=torch.float32)
    eigen_vals = torch.sum(eigenvecs * torch.matmul(logM, eigenvecs), dim=0)

    # Loss calculation
    loss = torch.sqrt(norm_logM - torch.sum(eigen_vals ** 2))

    # Backpropagation to get the gradient
    loss.backward()
    adj_matrix_grad = adj_matrix.grad.numpy()

    # Calculate gradient for each candidate edge
    row, col = candidates.T
    sig_est_grad = adj_matrix_grad[row, col] + adj_matrix_grad[col, row]
    ignore = sig_est_grad < 0
    sig_est_grad[ignore] = -1

    return sig_est_grad


@njit
def estimate_loss_with_delta_eigenvals(candidates, flip_indicator, vals_org,
                                       vecs_org, num_nodes, dim, window_size):
    """Computes the estimated loss using the change in the eigenvalues for every candidate edge flip.

    :param candidates: np.ndarray, shape [?,2]
        Candidate set of edge flips,
    :param flip_indicator: np.ndarray, shape [?]
        Vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    :param vals_org: np.ndarray, shape [n]
        The generalized eigenvalues of the clean graph
    :param vecs_org: np.ndarray, shape [n, n]
        The generalized eigenvectors of the clean graph
    :param num_nodes: int
        Number of nodes
    :param dim: int
        Embedding dimension
    :param window_size: int
        Size of the window
    :return: np.ndarray, shape [?]
        Estimated loss for each candidate flip
    """

    loss_est = np.zeros(candidates.shape[0])
    for x in range(candidates.shape[0]):
        i, j = candidates[x]
        vals_est = vals_org + flip_indicator[x] * (
            2 * vecs_org[i] * vecs_org[j] - vals_org *
            (vecs_org[i]**2 + vecs_org[j]**2))

        vals_sum_powers = sum_of_powers(vals_est, window_size)

        loss_ij = np.sqrt(np.sum(
            np.sort(vals_sum_powers**2)[:num_nodes - dim]))
        loss_est[x] = loss_ij

    return loss_est


@njit
def sum_of_powers(x, power):
    """For each x_i, computes \sum_{r=1}^{pow) x_i^r (elementwise sum of powers).

    :param x: shape [?]
        Any vector
    :param pow: int
        The largest power to consider
    :return: shape [?]
        Vector where each element is the sum of powers from 1 to pow.
    """
    n = x.shape[0]
    sum_powers = np.zeros((power, n))
    for i, i_power in enumerate(range(1, power + 1)):
        sum_powers[i] = np.power(x, i_power)

    return sum_powers.sum(0)