from pathlib import Path

# Path về folder gốc
ROOT = Path(__file__).resolve().parents[2]

# Path các file data/model/figures
RAW_PATH = ROOT / "data" / "raw" / "raw_viwiki.jsonl"
CLEAN_PATH = ROOT / "data" / "processed" / "corpus_clean.txt"
SEG_PATH = ROOT / "data" / "processed" / "corpus_seg.txt"
MODEL_PATH = ROOT / "models" / "w2v_vectors.npz"
FIG_DIR = ROOT / "figures"

# Tham số mặc định
CFG = {
    "dim": 100,               # Số chiều word vector
    "window": 5,              # Cửa sổ ngữ cảnh
    "negatives": 5,           # Số từ âm K cho mỗi cặp
    "batch": 4096,            
    "epochs": 10,
    "lr_max": 0.025,          # Learning rate cực đại ban đầu
    "lr_min": 0.0001,         # Learning rate tối thiểu ở epoch cuối
    "subsample_t": 1e-4,      
    "noise_alpha": 0.5,       
    "seed": 0,
}

MIN_COUNT = 5                 # Loại bỏ nếu từ xuất hiện < 5 lần do quá ít