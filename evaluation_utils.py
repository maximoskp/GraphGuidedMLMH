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
from train_utils import make_mixed_batch, full_to_partial_masking

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

# =========== Contrastive space convergence ==================

def evaluate_contrastive_convergence(
        transformer_model,
        contrastive_model,
        val_loader,
        logits_loss_fn,
        contrastive_loss_fn,
        source_key,
        mask_token_id,
        bar_token_id,
        device,
        interpolate=False,
        extrapolate=False
):
    tqdm_position = 0
    epoch = 0
    step = 0
    num_visible = 0

    with torch.no_grad():
        running_foreign_loss = 0
        val_foreign_loss = 0

        running_home_loss = 0
        val_home_loss = 0

        running_no2home_loss = 0
        val_no2home_loss = 0

        running_no2foreign_loss = 0
        val_no2foreign_loss = 0

        # logits losses
        running_foreign_logits_loss = 0
        val_foreign_logits_loss = 0
        running_home_logits_loss = 0
        val_home_logits_loss = 0
        running_no2home_logits_loss = 0
        val_no2home_logits_loss = 0

        # accuracies
        running_foreign_acc = 0
        val_foreign_acc = 0
        running_home_acc = 0
        val_home_acc = 0
        running_no2home_acc = 0
        val_no2home_acc = 0

        batch_num = 0
        print('validation')
        with tqdm(val_loader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch}@{step}| val')
            for batch in tepoch:
                batch_num += 1
                melody_grid = batch["pianoroll"].to(device)
                harmony_gt = batch["harmony_ids"].to(device)
                home_guidance_embeddings = batch[source_key].to(device)
                mixed_batch_1 = make_mixed_batch(batch, source_key)
                mixed_batch_2 = make_mixed_batch(mixed_batch_1, source_key)
                if interpolate:
                    mixed_batch_1 = make_mixed_batch(batch, source_key)
                    mixed_batch_2 = make_mixed_batch(mixed_batch_1, source_key)
                    foreign_guidance_embeddings_1 = mixed_batch_1[source_key].to(device)
                    foreign_guidance_embeddings_2 = mixed_batch_2[source_key].to(device)
                    z_guidance_1 = contrastive_model.source_proj(foreign_guidance_embeddings_1.to(device))
                    z_guidance_2 = contrastive_model.source_proj(foreign_guidance_embeddings_2.to(device))
                    z_foreign_guidance = 0.5*z_guidance_1 + 0.5*z_guidance_2
                elif extrapolate:
                    foreign_guidance_embeddings = mixed_batch_1[source_key].to(device)
                    z_foreign_guidance = 10*contrastive_model.source_proj(foreign_guidance_embeddings.to(device))
                else:
                    foreign_guidance_embeddings = mixed_batch_1[source_key].to(device)
                    z_foreign_guidance = contrastive_model.source_proj(foreign_guidance_embeddings.to(device))

                harmony_input, harmony_target = full_to_partial_masking(
                    harmony_gt,
                    mask_token_id,
                    num_visible,
                    bar_token_id=bar_token_id
                )

                # Step 1: contrastive latent attraction validation
                logits, hidden = transformer_model(
                    melody_grid.to(device),
                    harmony_input.to(device),
                    z_foreign_guidance.to(device),
                    return_hidden=True
                )
                # z_guidance, z_transformer, _ = contrastive_model( foreign_guidance_embeddings.to(device), hidden.to(device))
                z_transformer = contrastive_model.transformer_proj(hidden.to(device))
                foreign_guidance_loss = contrastive_loss_fn(z_foreign_guidance,z_transformer)

                running_foreign_logits_loss += logits_loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1)).item()
                val_foreign_logits_loss = running_foreign_logits_loss/batch_num

                # accuracy
                predictions = logits.argmax(dim=-1)
                # mask = torch.logical_and(harmony_target != harmony_input, harmony_target != -100)
                mask = harmony_target != -100
                running_foreign_acc += (predictions[mask] == harmony_target[mask]).sum().item()/mask.sum().item()
                val_foreign_acc = running_foreign_acc/batch_num

                # Step 2: home attraction validation
                z_guidance = contrastive_model.source_proj(home_guidance_embeddings.to(device))
                logits, hidden = transformer_model(
                    melody_grid.to(device),
                    harmony_input.to(device),
                    z_guidance.to(device),
                    return_hidden=True
                )
                z_guidance, z_transformer, _ = contrastive_model( home_guidance_embeddings.to(device), hidden.to(device))
                home_guidance_loss = contrastive_loss_fn(z_guidance,z_transformer)

                running_home_logits_loss += logits_loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1)).item()
                val_home_logits_loss = running_home_logits_loss/batch_num

                # accuracy
                predictions = logits.argmax(dim=-1)
                # mask = torch.logical_and(harmony_target != harmony_input, harmony_target != -100)
                mask = harmony_target != -100
                running_home_acc += (predictions[mask] == harmony_target[mask]).sum().item()/mask.sum().item()
                val_home_acc = running_home_acc/batch_num

                # partial losses
                running_foreign_loss += foreign_guidance_loss.item()
                val_foreign_loss = running_foreign_loss/batch_num
                running_home_loss += home_guidance_loss.item()
                val_home_loss = running_home_loss/batch_num

                # Step 3: no attraction
                logits, hidden = transformer_model(
                    melody_grid.to(device),
                    harmony_input.to(device),
                    None,
                    return_hidden=True
                )
                z_transformer = contrastive_model.transformer_proj(hidden.to(device))
                # z_guidance, z_transformer, _ = contrastive_model( home_guidance_embeddings.to(device), hidden.to(device))
                no_guidance_to_home_loss = contrastive_loss_fn(z_guidance,z_transformer)

                # no_guidance_to_home_logits_loss = logits_loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1))

                # z_guidance, z_transformer, _ = contrastive_model( foreign_guidance_embeddings.to(device), hidden.to(device))
                no_guidance_to_foreign_loss = contrastive_loss_fn(z_foreign_guidance,z_transformer)

                running_no2home_logits_loss += logits_loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1)).item()
                val_no2home_logits_loss = running_no2home_logits_loss/batch_num

                # accuracy
                predictions = logits.argmax(dim=-1)
                # mask = torch.logical_and(harmony_target != harmony_input, harmony_target != -100)
                mask = harmony_target != -100
                running_no2home_acc += (predictions[mask] == harmony_target[mask]).sum().item()/mask.sum().item()
                val_no2home_acc = running_no2home_acc/batch_num

                # partial losses
                running_no2home_loss += no_guidance_to_home_loss.item()
                val_no2home_loss = running_no2home_loss/batch_num
                running_no2foreign_loss += no_guidance_to_foreign_loss.item()
                val_no2foreign_loss = running_no2foreign_loss/batch_num

                loss = 0.25*home_guidance_loss + 0.25*foreign_guidance_loss + 0.25*no_guidance_to_home_loss + 0.25*no_guidance_to_foreign_loss
                acc = 0.5*val_home_acc + 0.5*val_no2home_acc

                tepoch.set_postfix(
                    loss=loss.item(),
                    floss=val_foreign_loss,
                    hloss=val_home_loss,
                    acc=acc
                )
            # end for batch
        # end with tqdm
    # end with no grad
    return {
        'val_foreign_loss': val_foreign_loss,
        'val_home_loss': val_home_loss,
        'val_no2home_loss': val_no2home_loss,
        'val_no2foreign_loss': val_no2foreign_loss,
        'val_home_logits_loss':  val_home_logits_loss,
        'val_foreign_logits_loss':  val_foreign_logits_loss,
        'val_no2home_logits_loss': val_no2home_logits_loss,
        'val_home_acc': val_home_acc,
        'val_foreign_acc': val_foreign_acc,
        'val_no2home_acc': val_no2home_acc
    }
#  end evaluate_contrastive_convergence