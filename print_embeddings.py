import numpy as np

from embedding_utils import DEFAULT_SENTENCES, get_embeddings


def format_vector(vector: np.ndarray, precision: int = 4) -> str:
    """
    np.array2string() のラッパー。桁数を整えて読みやすくする。
    """
    return np.array2string(
        vector,
        precision=precision,
        separator=", ",
        floatmode="fixed",
        suppress_small=False,
    )


def main() -> None:
    sentences = DEFAULT_SENTENCES
    embeddings = get_embeddings(sentences)

    print("サンプル文の埋め込みベクトル:\n")
    for idx, (sentence, vector) in enumerate(zip(sentences, embeddings)):
        print(f"[{idx}] {sentence}")
        print(format_vector(vector))
        print()


if __name__ == "__main__":
    main()
