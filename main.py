from embedding_utils import (
    DEFAULT_SENTENCES,
    cosine_similarity,
    get_embeddings,
)

# ==========
# 1. 対象となる文章（元の文＋10文）
# ==========
sentences = DEFAULT_SENTENCES

# ==========
# 2. 埋め込みを取得する関数
# ==========
# embedding_utils.get_embeddings() を利用

# ==========
# 3. コサイン類似度を計算する関数
# ==========
# embedding_utils.cosine_similarity() を利用

# ==========
# 4. 実行：埋め込み取得 ＋ 類似度計算
# ==========
embs = get_embeddings(sentences)

# 元の文（インデックス 0）
base_vec = embs[0]

results = []
for idx in range(1, len(sentences)):
    sim = cosine_similarity(base_vec, embs[idx])
    results.append((idx, sentences[idx], sim))

# 類似度の高い順にソート
results_sorted = sorted(results, key=lambda x: x[2], reverse=True)

# ==========
# 5. 結果表示
# ==========
print("元の文:")
print(sentences[0])
print("\n他の文とのコサイン類似度:\n")

for idx, sent, sim in results_sorted:
    print(f"[{idx}] 類似度={sim:.3f} : {sent}")
