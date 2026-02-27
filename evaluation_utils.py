from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models import ContrastiveSpaceModel
import pickle
import numpy as np
import os
from train_utils import train_contrastive
from data_utils import ContrastiveCollator

def collect_embeddings(model, dataloader, source_key, device):
    model.eval()
    
    all_z_s = []
    all_z_t = []

    with torch.no_grad():
        for batch in dataloader:
            source_emb = batch[source_key].to(device)
            transformer_emb = batch['transformer_embeddings'].to(device)

            z_s, z_t, temp = model(source_emb, transformer_emb)

            all_z_s.append(z_s)
            all_z_t.append(z_t)

    Zs = torch.cat(all_z_s, dim=0)
    Zt = torch.cat(all_z_t, dim=0)

    return Zs, Zt
# end collect_embeddings

def procrustes_error(Zs, Zt):
    """
    Computes orthogonal Procrustes alignment error.
    Returns Frobenius norm error and aligned embeddings.
    """

    # Center both spaces
    Zs_centered = Zs - Zs.mean(dim=0, keepdim=True)
    Zt_centered = Zt - Zt.mean(dim=0, keepdim=True)

    # Compute cross-covariance
    M = Zs_centered.T @ Zt_centered

    # SVD
    U, _, Vt = torch.linalg.svd(M)
    R = U @ Vt

    # Align Zs
    Zs_aligned = Zs_centered @ R

    # Frobenius norm error
    error = torch.norm(Zs_aligned - Zt_centered, p='fro')

    return error.item(), Zs_aligned, Zt_centered
# end procrustes_error

def distance_matrix(Z):
    return torch.cdist(Z, Z, p=2)

def distance_matrix_correlation(Zs, Zt):
    Ds = distance_matrix(Zs)
    Dt = distance_matrix(Zt)

    # Flatten upper triangular (without diagonal)
    idx = torch.triu_indices(Ds.size(0), Ds.size(1), offset=1)

    ds_flat = Ds[idx[0], idx[1]]
    dt_flat = Dt[idx[0], idx[1]]

    # Pearson correlation
    corr = torch.corrcoef(torch.stack([ds_flat, dt_flat]))[0,1]

    return corr.item()
# end distance_matrix_correlation

def retrieval_accuracy(Zs, Zt, topk=(1,5)):
    """
    Computes cross-space retrieval accuracy.
    """

    # Normalize (important!)
    Zs = F.normalize(Zs, dim=1)
    Zt = F.normalize(Zt, dim=1)

    # Similarity matrix
    sim = Zt @ Zs.T   # [N, N]

    results = {}

    for k in topk:
        correct = 0
        topk_indices = sim.topk(k, dim=1).indices

        for i in range(sim.size(0)):
            if i in topk_indices[i]:
                correct += 1

        results[f"top{k}"] = correct / sim.size(0)

    return results
# end retrieval_accuracy

def evaluate_alignment(model, dataloader, source_key, device):

    Zs, Zt = collect_embeddings(model, dataloader, source_key, device)

    proc_error, _, _ = procrustes_error(Zs, Zt)
    rsa_corr = distance_matrix_correlation(Zs, Zt)
    retrieval = retrieval_accuracy(Zs, Zt)

    print("=== Alignment Evaluation ===")
    print("Procrustes error:", proc_error)
    print("Distance matrix correlation:", rsa_corr)
    print("Retrieval accuracy:", retrieval)

    return {
        "procrustes_error": proc_error,
        "rsa_correlation": rsa_corr,
        "retrieval": retrieval
    }
# end evaluate_alignment