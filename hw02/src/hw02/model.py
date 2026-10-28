import numpy as np

def init_weights(vocab_size, dim, seed=0, dtype=np.float32):
    """Khởi tạo ma trận trọng số W_in và W_out."""
    rng = np.random.default_rng(seed)
    W_in = ((rng.random((vocab_size, dim)) - 0.5) / dim).astype(dtype)
    W_out = np.zeros((vocab_size, dim), dtype=dtype)
    return W_in, W_out

def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))

def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)

def compute_loss_and_grads(W_in, W_out, centers, contexts, negatives):
    v = W_in[centers]               # (B, D)
    u_pos = W_out[contexts]         # (B, D)
    u_neg = W_out[negatives]        # (B, K, D)

    scores_pos = np.sum(v * u_pos, axis=1)                          # (B,)
    scores_neg = np.sum(v[:, None, :] * u_neg, axis=-1)             # (B, K)

    loss = -np.sum(log_sigmoid(scores_pos)) - np.sum(log_sigmoid(-scores_neg))

    g_pos = sigmoid(scores_pos) - 1.0                              # (B,)
    g_neg = sigmoid(scores_neg)                                     # (B, K)

    grad_v = g_pos[:, None] * u_pos + np.sum(g_neg[:, :, None] * u_neg, axis=1)
    grad_u_pos = g_pos[:, None] * v
    grad_u_neg = g_neg[:, :, None] * v[:, None, :]

    return float(loss), grad_v, grad_u_pos, grad_u_neg

def sgd_step_np(W_in, W_out, centers, contexts, negatives, lr):
    loss, grad_v, grad_u_pos, grad_u_neg = compute_loss_and_grads(W_in, W_out, centers, contexts, negatives)
    dim = W_in.shape[1]
    
    np.add.at(W_in, centers, -lr * grad_v)
    np.add.at(W_out, contexts, -lr * grad_u_pos)
    np.add.at(W_out, negatives.reshape(-1), -lr * grad_u_neg.reshape(-1, dim))
    return loss