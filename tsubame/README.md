# TSUBAME 4.0 での実行方法

TSUBAME 4.0 (東工大) で llm-jp-eval-mm の評価を PBS ジョブとして実行するためのスクリプト群です。

## 前提

- PBS Pro (qsub) によるジョブスケジューリング
- H100 x 4 / ノード
- グループストレージ: `/gs/bs/<group>/`
- Python 環境: uv

## ファイル構成

```
tsubame/
├── config.sh       # 共通設定（パス、モデルリスト、タスク定義）
├── run_model.sh    # PBS ジョブスクリプト（1モデル = 1ジョブ）
├── submit_all.sh   # 一括投入ヘルパー
└── README.md       # このファイル
```

## セットアップ

### 1. config.sh の編集

`config.sh` 内の以下の変数を環境に合わせて編集してください:

```bash
TSUBAME_GROUP="tga-NII-LLM"          # PBS グループ名
TSUBAME_QUEUE="gpu_h100"              # PBS キュー名
PROJECT_DIR="/gs/bs/.../eval-mm"      # プロジェクトのクローン先
```

環境変数で上書きすることもできます:

```bash
export TSUBAME_GROUP="your-group"
export PROJECT_DIR="/gs/bs/your-group/eval-mm"
```

### 2. プロジェクトのクローンと初期設定

```bash
cd /gs/bs/<group>/
git clone https://github.com/llm-jp/llm-jp-eval-mm.git eval-mm
cd eval-mm

# uv のインストール（未導入の場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# .env に API キーを設定
cp .env.example .env  # あれば
# HF_TOKEN, OPENAI_API_KEY 等を記入
```

## 使い方

### 全モデルを一括投入

```bash
bash tsubame/submit_all.sh
```

### dry-run（投入せず確認）

```bash
bash tsubame/submit_all.sh --dry-run
```

### バックエンドで絞り込み

```bash
bash tsubame/submit_all.sh --filter vllm           # vLLM モデルのみ
bash tsubame/submit_all.sh --filter transformers    # transformers モデルのみ
```

### モデル名で絞り込み

```bash
bash tsubame/submit_all.sh --grep "Qwen"           # Qwen 系のみ
bash tsubame/submit_all.sh --grep "InternVL3"       # InternVL3 系のみ
```

### 単一モデルを手動投入

```bash
qsub -v MODEL_ENTRY="Qwen/Qwen2.5-VL-7B-Instruct|vllm_normal|vllm|1" tsubame/run_model.sh
```

## ジョブの確認

```bash
qstat -u $USER              # 自分のジョブ一覧
qstat -f <job_id>           # ジョブ詳細
qdel <job_id>               # ジョブ削除
```

ログは `$LOG_DIR`（デフォルト: `/gs/bs/<group>/eval-mm-logs/`）に出力されます。

## カスタマイズ

### 実行時間の変更

デフォルトは 24 時間です。小さいモデルのみなら短縮できます:

```bash
export TSUBAME_WALLTIME="6:00:00"
bash tsubame/submit_all.sh
```

### モデルリストの編集

`config.sh` 内の `MODEL_LIST` をコメントアウト/編集して、実行対象を調整してください。

### PBS キュー・リソースの変更

`run_model.sh` 先頭の `#PBS` ディレクティブを直接編集するか、`qsub` の引数で上書きしてください:

```bash
qsub -q other_queue -l walltime=12:00:00 -v MODEL_ENTRY="..." tsubame/run_model.sh
```

## 注意事項

- API ベースのモデル（gpt-4o 等）は計算ノードからインターネットアクセスできない場合があるため、デフォルトでコメントアウトしています
- 初回実行時はモデルのダウンロードに時間がかかります。事前にログインノードで `huggingface-cli download <model_id>` しておくことを推奨します
- 結果は `$RESULT_DIR` に保存されます（モデル別のサブディレクトリ）
