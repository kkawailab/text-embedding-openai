# text-embedding-openai

## 概要
`main.py` は OpenAI の最新埋め込みモデル（デフォルトで `text-embedding-3-small`）を使って、指定した 11 文の意味的な近さを比較するシンプルなデモです。最初の文を基準ベクトルにし、残りの文とのコサイン類似度を計算して高い順にコンソールへ表示します。なお、初めてこのリポジトリに触れる方や初心者は、セットアップの手順と概念の流れをまとめた `intro.md` を先に読むようにしてください。

## セットアップ
1. Python 3.13 以上とパッケージマネージャー `uv` を用意します（まだの場合は `pip install uv` などで導入）。
2. プロジェクトルートで依存関係を同期します。
   ```bash
   uv sync
   ```
3. OpenAI の API キーを環境変数に設定します（`.env` への書き込みは禁止されているので、シェルの環境で設定してください）。
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
4. 動作確認として後述の `uv run python main.py` を実行します。CLI で 11 文の類似度が表示されれば準備完了です。

詳しい背景説明や、各ステップの意味を初心者向けに噛み砕いた解説は `intro.md` にまとまっています。

## 依存関係と環境
- Python 3.13 以上
- `uv` で解決される `numpy` と `openai` ライブラリ（`pyproject.toml` を参照）
- 環境変数 `OPENAI_API_KEY` に OpenAI API キーを設定しておく必要があります。

## 実行方法
プロジェクトルートで以下を実行すると、埋め込み取得から類似度算出、結果表示までを一度に行います。

```bash
uv run python main.py
```

スクリプトは `get_embeddings()` で文章リストを API に送り、返ってきたベクトルを `numpy` 配列に格納します。その後 `cosine_similarity()` を使って基準文と他文の距離を計算し、似ている順に `[index] 類似度=... : 文` という形式で出力します。

## cosine_similarity_basics.py（教育用の最小ステップ）
`cosine_similarity_basics.py` は `main.py` よりさらにストレートな「1 ファイルで完結する」教材用スクリプトです。環境変数の読み込みから API 呼び出し、コサイン類似度の算出までを順番に追いながら学習できるよう、各ステップに丁寧なコメントを入れています。

```bash
uv run python cosine_similarity_basics.py
```

主なポイントは次のとおりです。
- 先頭の文（インデックス 0）と残り 10 文を比較する流れをそのまま記述
- `OPENAI_API_KEY` が設定されていない場合はすぐに終了して注意喚起
- `get_embeddings` / `cosine_similarity` の役割をコメントで詳細に解説し、授業中に読み上げながら進めやすい構成

埋め込みベクトルそのものを確認したい場合は、以下の補助スクリプトを利用できます。

```bash
uv run python print_embeddings.py
```

各文の直後に 1 行の埋め込みベクトル（`numpy.array2string` 形式）が表示されるので、値をコピペしたり、別ツールに貼り付けて解析することができます。`embedding_utils.py` の `DEFAULT_SENTENCES` を書き換えれば任意の文集合にも対応します。

## 応用のヒント
- モデル名（`model` 引数）を `text-embedding-3-large` などに変更可能です。
- `sentences` リストを入れ替えるだけで任意の文集合に対してランキングを取得できます。
- 類似度のしきい値を導入すれば、しきい値以上だけを抽出するフィルタリングのベースとして再利用できます。

## 更新履歴
- （未リリース）初心者向けに `intro.md` を読む導線を README に追加
- `86e87fa` Rename educational cosine script — 教材向けスクリプトのファイル名を分かりやすく整理
- `a465508` Add beginner-focused cosine similarity script — ステップバイステップで学べる教育用スクリプトを追加
- `596228e` Add embedding utilities and docs — 共通ユーティリティと README ドキュメントを拡充
- `11be665` Add initial project files and configurations — 初期セットアップと主要スクリプトを配置
