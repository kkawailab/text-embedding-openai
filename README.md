# text-embedding-openai

## 概要
`main.py` は OpenAI の最新埋め込みモデル（デフォルトで `text-embedding-3-small`）を使って、指定した 11 文の意味的な近さを比較するシンプルなデモです。最初の文を基準ベクトルにし、残りの文とのコサイン類似度を計算して高い順にコンソールへ表示します。

## 依存関係と環境
- Python 3.13 以上
- `uv` で解決される `numpy` と `openai` ライブラリ（`pyproject.toml` を参照）
- 環境変数 `OPENAI_API_KEY` に OpenAI API キーを設定しておく必要があります。

```bash
export OPENAI_API_KEY="sk-..."
uv sync  # 依存関係のインストール
```

## 実行方法
プロジェクトルートで以下を実行すると、埋め込み取得から類似度算出、結果表示までを一度に行います。

```bash
uv run python main.py
```

スクリプトは `get_embeddings()` で文章リストを API に送り、返ってきたベクトルを `numpy` 配列に格納します。その後 `cosine_similarity()` を使って基準文と他文の距離を計算し、似ている順に `[index] 類似度=... : 文` という形式で出力します。

埋め込みベクトルそのものを確認したい場合は、以下の補助スクリプトを利用できます。

```bash
uv run python print_embeddings.py
```

各文の直後に 1 行の埋め込みベクトル（`numpy.array2string` 形式）が表示されるので、値をコピペしたり、別ツールに貼り付けて解析することができます。`embedding_utils.py` の `DEFAULT_SENTENCES` を書き換えれば任意の文集合にも対応します。

## 応用のヒント
- モデル名（`model` 引数）を `text-embedding-3-large` などに変更可能です。
- `sentences` リストを入れ替えるだけで任意の文集合に対してランキングを取得できます。
- 類似度のしきい値を導入すれば、しきい値以上だけを抽出するフィルタリングのベースとして再利用できます。
