import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CFG, MODEL_PATH, FIG_DIR, SEG_PATH
from data import load_dataset, calculate_subsample_probs, extract_skipgram_pairs, build_noise_distribution
from model import init_weights

def sgd_step_gpu(W_in, W_out, centers, contexts, negatives, lr):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        dim = W_in.shape[1]
        v = W_in[centers]
        u_pos = W_out[contexts]
        u_neg = W_out[negatives]

        scores_pos = (v * u_pos).sum(dim=1)
        scores_neg = (v.unsqueeze(1) * u_neg).sum(dim=-1)

        loss = -(F.logsigmoid(scores_pos).sum() + F.logsigmoid(-scores_neg).sum())

        g_pos = torch.sigmoid(scores_pos) - 1.0
        g_neg = torch.sigmoid(scores_neg)

        grad_v = g_pos[:, None] * u_pos + (g_neg[:, :, None] * u_neg).sum(dim=1)
        grad_u_pos = g_pos[:, None] * v
        grad_u_neg = g_neg[:, :, None] * v[:, None, :]

        W_in.index_add_(0, centers, grad_v, alpha=-lr)
        W_out.index_add_(0, contexts, grad_u_pos, alpha=-lr)
        W_out.index_add_(0, negatives.reshape(-1), grad_u_neg.reshape(-1, dim), alpha=-lr)
        return loss

def sample_negatives(cdf_tensor, batch_size, k):
    import torch
    u = torch.rand(batch_size, k, device=cdf_tensor.device, dtype=cdf_tensor.dtype)
    return torch.searchsorted(cdf_tensor, u, right=True).clamp_(max=cdf_tensor.numel() - 1)

def save_model_checkpoint(path, W_in, W_out, vocab, counts):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, W_in=W_in, W_out=W_out, vocab=np.array(vocab), counts=counts)

def train_model():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang train bằng: {device}")

    vocab, word2id, counts, ids, sids = load_dataset()
    rng = np.random.default_rng(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    keep_probs = calculate_subsample_probs(counts, t=CFG["subsample_t"])
    noise_dist = build_noise_distribution(counts, alpha=CFG["noise_alpha"])

    W_in_np, W_out_np = init_weights(len(vocab), CFG["dim"], seed=CFG["seed"])
    W_in = torch.from_numpy(W_in_np).to(device)
    W_out = torch.from_numpy(W_out_np).to(device)
    cdf_tensor = torch.from_numpy(np.cumsum(noise_dist)).to(device=device, dtype=torch.float32)

    loss_history = []
    t_start = time.time()

    for epoch in range(CFG["epochs"]):
        t0 = time.time()
        
        mask = rng.random(len(ids)) < keep_probs[ids]
        c_np, o_np = extract_skipgram_pairs(ids[mask], sids[mask], CFG["window"])
        
        centers = torch.from_numpy(c_np).to(device)
        contexts = torch.from_numpy(o_np).to(device)
        
        perm = torch.randperm(len(centers), device=device)
        centers, contexts = centers[perm], contexts[perm]

        num_pairs = len(centers)
        num_batches = (num_pairs + CFG["batch"] - 1) // CFG["batch"]
        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)

        for b in range(num_batches):
            progress = (epoch + b / num_batches) / CFG["epochs"]
            lr = CFG["lr_min"] + 0.5 * (CFG["lr_max"] - CFG["lr_min"]) * (1.0 + math.cos(math.pi * progress))

            batch_slice = slice(b * CFG["batch"], (b + 1) * CFG["batch"])
            batch_c, batch_o = centers[batch_slice], contexts[batch_slice]
            batch_neg = sample_negatives(cdf_tensor, len(batch_c), CFG["negatives"])

            epoch_loss_sum += sgd_step_gpu(W_in, W_out, batch_c, batch_o, batch_neg, lr)

        mean_loss = epoch_loss_sum.item() / num_pairs
        loss_history.append(mean_loss)
        print(f"Epoch {epoch + 1:2d}/{CFG['epochs']} | {num_pairs:,} cặp | Loss/cặp = {mean_loss:.4f} | LR = {lr:.5f} | {time.time() - t0:.1f}s")
        
        save_model_checkpoint(MODEL_PATH, W_in.cpu().numpy(), W_out.cpu().numpy(), vocab, counts)

    print(f"Hoàn thành huấn luyện trong {time.time() - t_start:.1f}s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    Path(MODEL_PATH).with_name("train_log.json").write_text(
        json.dumps({"cfg": CFG, "loss_history": loss_history, "vocab_size": len(vocab)}, indent=2), encoding="utf-8"
    )
    return loss_history

if __name__ == "__main__":
    history = train_model()