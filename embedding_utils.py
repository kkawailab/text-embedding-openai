import os
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

__all__ = [
    "DEFAULT_SENTENCES",
    "get_embeddings",
    "cosine_similarity",
]


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# サンプルとして使っている文章一覧（0番目が基準文）
DEFAULT_SENTENCES: list[str] = [
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


def get_embeddings(texts: Sequence[str]) -> NDArray[np.float32]:
    """
    texts: str のシーケンス
    return: np.ndarray (len(texts), embedding_dim)
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """
    a, b: 1次元ベクトル
    return: cos 類似度（スカラー）
    """
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
