import os
import numpy as np
from openai import OpenAI

# ==========
# 0. OpenAI クライアント設定
# ==========
# 環境変数 OPENAI_API_KEY に API キーを入れておくのがおすすめです
#   export OPENAI_API_KEY="sk-...."
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ==========
# 1. 対象となる文章（元の文＋10文）
# ==========
sentences = [
    "今朝、わたしはとても機嫌よく起床しました。",  # 0: 元の文
    "今朝は、気分爽快で目を覚ますことができました。",                 # 1: A
    "今朝の目覚めはとてもよく、すっきりと起きられました。",             # 2: B
    "今日は朝から気持ちよく起床できました。",                       # 3: C
    "目が覚めた瞬間から、今朝はとても晴れやかな気分でした。",         # 4: D
    "今朝は、心が軽くなるような良い気分で起きました。",               # 5: E
    "今日は、いつもより機嫌よく朝を迎えました。",                   # 6: F
    "今朝の私は、とても清々しい気持ちで布団から出られました。",       # 7: G
    "朝起きたとき、自然と笑顔になるくらい気分がよかったです。",       # 8: H
    "今朝は、とても快適な気分で目を覚ましました。",                 # 9: I
    "今日は、気持ちが明るいまま心地よく起床しました。",             # 10: J
]

# ==========
# 2. 埋め込みを取得する関数
# ==========
def get_embeddings(texts):
    """
    texts: str のリスト
    return: np.ndarray (len(texts), embedding_dim)
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",   # お好みで "text-embedding-3-large" など
        input=texts
    )
    # response.data[i].embedding にベクトルが入っている
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")

# ==========
# 3. コサイン類似度を計算する関数
# ==========
def cosine_similarity(a, b):
    """
    a, b: 1次元ベクトル (np.ndarray)
    return: cos 類似度（スカラー）
    """
    # 0除算を避けるために少しだけ安全策
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

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
