import numpy as np

from config import SEG_PATH, MIN_COUNT
from collections import Counter


def build_vocab(path, min_count=5):
    """Đếm tần suất, lọc bỏ từ hiếm, rồi sắp xếp các từ theo số lần xuất hiện giảm dần"""
    counter = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            counter.update(line.split())
            
    kept_items = [(word, cnt) for word, cnt in counter.most_common() if cnt >= min_count]
    vocab = [word for word, _ in kept_items]
    word2id = {word: idx for idx, word in enumerate(vocab)}
    counts = np.array([cnt for _, cnt in kept_items], dtype=np.float64)
    return vocab, word2id, counts

def encode_corpus(path, word2id):
    """Đưa văn bản thành mảng id các từ và id trong câu câu tương ứng"""
    token_list, sent_list = [], []
    with open(path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            for token in line.split():
                word_id = word2id.get(token)
                if word_id is not None:
                    token_list.append(word_id)
                    sent_list.append(line_idx)
                    
    return np.array(token_list, dtype=np.int64), np.array(sent_list, dtype=np.int64)

def calculate_subsample_probs(counts, t=1e-4):
    freqs = counts / counts.sum()
    keep_probs = np.sqrt(t / freqs)
    return np.minimum(1.0, keep_probs)

def extract_skipgram_pairs(ids, sids, window_size):
    """Sinh các cặp (trung tâm, ngữ cảnh) nằm trong cùng câu"""
    centers, contexts = [], []
    num_tokens = len(ids)
    
    for offset in range(1, window_size + 1):
        if offset >= num_tokens:
            break
        same_sent = (sids[:-offset] == sids[offset:])
        left_ids = ids[:-offset][same_sent]
        right_ids = ids[offset:][same_sent]
        
        centers.extend([left_ids, right_ids])
        contexts.extend([right_ids, left_ids])
        
    if not centers:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
        
    return np.concatenate(centers), np.concatenate(contexts)

def build_noise_distribution(counts, alpha=0.5):
    """Phân phối lấy mẫu từ âm với số mũ alpha"""
    weights = counts ** alpha
    return weights / weights.sum()

def load_dataset(limit=None):
    vocab, word2id, counts = build_vocab(SEG_PATH, MIN_COUNT)
    ids, sids = encode_corpus(SEG_PATH, word2id)
    if limit:
        ids, sids = ids[:limit], sids[:limit]
    print(f"Tổng từ vựng: {len(vocab):,} từ | Tổng số tokens: {len(ids):,}")
    return vocab, word2id, counts, ids, sids