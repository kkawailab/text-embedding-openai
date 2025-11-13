"""simple_calc_cos.py
初心者向けに、OpenAI の埋め込み API でテキストの類似度を計算する最低限の流れを示すサンプル。

main.py よりさらにストレートな構成にしてあり、各ステップをコメントで丁寧に説明しています。
"""

import os
import sys

import numpy as np
from openai import OpenAI


# ==========
# 0. OpenAI クライアント設定
# ==========
# (1) まずは環境変数から API キーを取り出す。初心者は "export OPENAI_API_KEY=..." をターミナルで実行しておく。
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    sys.exit(
        "OPENAI_API_KEY が設定されていません。`export OPENAI_API_KEY=...` を実行してから再試行してください。"
    )

# (2) 取り出したキーを使ってクライアントを作成。これ以降は client が API への窓口になる。
client = OpenAI(api_key=api_key)


# ==========
# 1. 対象となる文章（元の文＋10文）
# ==========
# 11 文のリストを用意。インデックス 0 の文を「基準」にして、残り 10 文との近さを測る。
sentences = [
    "今朝、わたしはとても機嫌よく起床しました。",  # 0: 元の文（比較の基準）
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
    """文章リストを OpenAI API に送り、各文章の埋め込みベクトルを取得する。"""

    # API に渡す引数は model（使いたいモデル名）と input（文字列の配列）の 2 つだけ。
    response = client.embeddings.create(
        model="text-embedding-3-small",  # 学習用には軽量な small モデルで十分。
        input=texts,
    )

    # response.data[i].embedding に実数のリスト（ベクトル）が入っているので、
    # それを順番に numpy の配列へ変換して返す。
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


# ==========
# 3. コサイン類似度を計算する関数
# ==========
def cosine_similarity(a, b):
    """2 つのベクトルがどれくらい似ているかを示すコサイン類似度を返す。"""

    # コサイン類似度 = (a・b) / (||a|| * ||b||)
    # ベクトルの長さが 0 だった場合は 0 除算を避けるために 0.0 を返す。
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ==========
# 4. 実行：埋め込み取得 ＋ 類似度計算
# ==========
# (1) sentences を API に送ってベクトルを作る。
embs = get_embeddings(sentences)

# (2) 先頭のベクトル（基準文）と残りのベクトルを 1 つずつ比較。
base_vec = embs[0]
results = []

for idx in range(1, len(sentences)):
    # enumerate を使っても良いが、ここではインデックスを明示して「0 とその他」を強調。
    sim = cosine_similarity(base_vec, embs[idx])
    results.append((idx, sentences[idx], sim))

# (3) 類似度（配列の 3 番目）で降順ソートし、似ている文から順に並べる。
results_sorted = sorted(results, key=lambda x: x[2], reverse=True)


# ==========
# 5. 結果表示
# ==========
print("元の文:")
print(sentences[0])
print("\n他の文とのコサイン類似度:\n")

for idx, sent, sim in results_sorted:
    # : .3f とすることで、類似度を小数第 3 位まで表示。
    print(f"[{idx}] 類似度={sim:.3f} : {sent}")


# ここから下は「ベタ書きのスクリプトがどのように実行順序を持つか」を説明するためのコメント。
# Python ではファイルの先頭から順番にコードが評価されるので、
# - 関数定義 (get_embeddings, cosine_similarity) はまず読み込まれ、
# - その後に実際の処理 (埋め込み取得→類似度計算→表示) が走ります。
# if __name__ == "__main__" ブロックを使う書き方もありますが、
# 教材としては「1 ファイルの流れ」をそのまま追えるよう、ここでは採用していません。
