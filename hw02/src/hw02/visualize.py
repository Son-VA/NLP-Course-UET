import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import MODEL_PATH, FIG_DIR

PAIRS = [
    ("ô_tô", "xe_máy"), ("máy_bay", "phi_cơ"), ("bác_sĩ", "y_tá"), ("to", "lớn"),
    ("thủ_đô", "hà_nội"), ("sông", "suối"), ("nóng", "lạnh"), ("tốt", "xấu"),
    ("chiến_tranh", "hòa_bình"), ("vua", "vương"),
]
TARGET_WORDS = [word for pair in PAIRS for word in pair]
MIN_WORD_COUNT = 50

def load_vectors(path):
    """Đọc W_in và chuẩn hóa vector về độ dài 1."""
    data = np.load(path)
    vocab = [str(w) for w in data["vocab"]]
    counts = data["counts"]
    W = data["W_in"]
    
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    unit_vectors = W / np.maximum(norms, 1e-12)
    word2id = {w: i for i, w in enumerate(vocab)}
    return word2id, counts, unit_vectors

def compute_pca(fit_data, transform_data, k_components=3):
    """Tính toán PCA bằng SVD trên toàn bộ từ rồi chiếu transform_data lên."""
    mean_vec = fit_data.mean(axis=0)
    centered_fit = fit_data - mean_vec
    
    _, S, Vt = np.linalg.svd(centered_fit, full_matrices=False)
    variance_ratio = (S ** 2) / np.sum(S ** 2)
    
    components = Vt[:k_components].copy()
    for i in range(k_components):
        max_idx = np.argmax(np.abs(components[i]))
        if components[i][max_idx] < 0:
            components[i] = -components[i]
            
    projected_coords = (transform_data - mean_vec) @ components.T
    return projected_coords, variance_ratio[:k_components]

def plot_pca_projection(coords, title, axis_labels, save_path):
    num_dim = coords.shape[1]
    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(projection="3d" if num_dim == 3 else None)
    
    for i, (word_a, word_b) in enumerate(PAIRS):
        p_a, p_b = coords[2 * i], coords[2 * i + 1]
        color = plt.cm.tab10(i)
        
        ax.plot(*zip(p_a, p_b), color=color, alpha=0.6, label=f"{word_a} – {word_b}")
        ax.scatter(*zip(p_a, p_b), color=color, s=50)
        ax.text(*p_a, f" {word_a}", color=color, fontsize=10)
        ax.text(*p_b, f" {word_b}", color=color, fontsize=10)
        
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    if num_dim == 3:
        ax.set_zlabel(axis_labels[2])
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.05, 0.5))
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def main():
    word2id, counts, unit_vectors = load_vectors(MODEL_PATH)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    missing = [w for w in TARGET_WORDS if w not in word2id]
    if missing:
        print("Có từ không tồn tại trong từ vựng:", missing)
        return

    # 1. Tính Cosine Similarity từng cặp
    print("===== ĐỘ TƯƠNG ĐỒNG COSINE VÀ CÁC CẶP TỪ =====")
    results_rows = []
    for a, b in PAIRS:
        id_a, id_b = word2id[a], word2id[b]
        cos_sim = float(np.dot(unit_vectors[id_a], unit_vectors[id_b]))
        results_rows.append({"tu_a": a, "tu_b": b, "cosine": round(cos_sim, 4)})
        print(f"{a:>12s} – {b:<12s} | Cosine: {cos_sim:.4f}")

    # 2. So sánh Cosine trung bình giữa cặp đôi vs khác cặp
    target_vectors = np.stack([unit_vectors[word2id[w]] for w in TARGET_WORDS])
    sim_matrix = target_vectors @ target_vectors.T
    
    pair_sims = [sim_matrix[2 * i, 2 * i + 1] for i in range(10)]
    cross_sims = [sim_matrix[i, j] for i in range(20) for j in range(i + 1, 20) if i // 2 != j // 2]
    print(f"\nCosine trung bình: Trong cặp đôi = {np.mean(pair_sims):.4f} | Giữa các từ khác cặp = {np.mean(cross_sims):.4f}")

    # 3. PCA để giảm chiều vector
    frequent_vocab_mask = counts >= MIN_WORD_COUNT
    fit_data_all = unit_vectors[frequent_vocab_mask]
    
    coords_3d, ratios = compute_pca(fit_data_all, target_vectors, k_components=3)
    
    print(f"\nPCA fit trên toàn bộ từ vựng ({int(frequent_vocab_mask.sum()):,} từ):")
    print(f"PC1 = {ratios[0]:.1%}, PC2 = {ratios[1]:.1%}, PC3 = {ratios[2]:.1%}")
    print(f"2D giữ {np.sum(ratios[:2]):.1%} phương sai | 3D giữ {np.sum(ratios):.1%} phương sai")

    axis_labels = [f"PC{i + 1} ({ratios[i]:.1%})" for i in range(3)]
    
    # Vẽ PCA 2D & 3D
    plot_pca_projection(
        coords_3d[:, :2], 
        f"Biểu đồ PCA 2D\nPhương sai giữ lại: {np.sum(ratios[:2]):.1%}",
        axis_labels[:2], 
        FIG_DIR / "pca2d_all.png"
    )
    plot_pca_projection(
        coords_3d, 
        f"Biểu đồ PCA 3D\nPhương sai giữ lại: {np.sum(ratios):.1%}",
        axis_labels, 
        FIG_DIR / "pca3d_all.png"
    )
    print("Đã xuất kết quả vào thư mục:", FIG_DIR)

if __name__ == "__main__":
    main()