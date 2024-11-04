import numpy as np
import torch

def neighbor_sampler_cpu(rowptr, col, idx, num_neighbors, replace=True):
    """
    Python implementation of neighbor sampler. 

    Parameters:
    -----------
    rowptr : torch.Tensor
        Row pointers of the sparse adjacency matrix in CSR format.
    col : torch.Tensor
        Column indices of the sparse adjacency matrix in CSR format.
    idx : torch.Tensor
        Nodes to sample neighbors from.
    num_neighbors : int
        Number of neighbors to sample per node. If -1, return all neighbors.
    replace : bool, optional
        Whether to sample with replacement. Default is True.

    Returns:
    --------
    targets : torch.Tensor
        Sampled target nodes (same as input idx).
    neighbors : torch.Tensor
        Sampled neighbors for each node in idx.
    e_id : torch.Tensor
        Edge indices corresponding to sampled neighbors.
    """
    rowptr_data = rowptr.numpy()  # Convert to numpy for easier processing
    col_data = col.numpy()
    idx_data = idx.numpy()

    targets = []
    neighbors = []
    e_id = []

    # Loop through each node in the input idx
    for n in idx_data:
        row_start = rowptr_data[n]
        row_end = rowptr_data[n + 1]
        row_count = row_end - row_start  # Number of neighbors for this node

        # If num_neighbors is -1, return all neighbors
        if num_neighbors < 0:
            perm = np.arange(row_count)
        else:
            if replace:
                # Sample with replacement
                perm = np.random.choice(row_count, num_neighbors, replace=True)
            else:
                # Sample without replacement
                perm = np.random.choice(row_count, min(num_neighbors, row_count), replace=False)

        # For each sampled neighbor, store the corresponding information
        for p in perm:
            e = row_start + p
            c = col_data[e]
            targets.append(n)
            neighbors.append(c)
            e_id.append(e)

    # Convert results back to torch tensors
    targets = torch.tensor(targets, dtype=torch.long)
    neighbors = torch.tensor(neighbors, dtype=torch.long)
    e_id = torch.tensor(e_id, dtype=torch.long)

    return targets, neighbors, e_id
