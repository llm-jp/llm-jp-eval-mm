# TSUBAME 4.0 での実行方法

TSUBAME 4.0 (東工大) で llm-jp-eval-mm の評価を PBS ジョブとして実行するためのスクリプト群です。

## 前提

- PBS Pro (qsub) によるジョブスケジューリング
- H100 x 4 / ノード
- Python 環境: uv

## ファイル構成

```
tsubame/
├── config.sh       # 共通定義（モデルリスト、タスク、メトリクス）
├── run_model.sh    # PBS ジョブスクリプト（1モデル = 1ジョブ）
├── submit_all.sh   # 一括投入ヘルパー
└── README.md       # このファイル
```

## セットアップ

### 1. プロジェクトのクローン

```bash
cd /gs/bs/<group>/<user>/
git clone https://github.com/llm-jp/llm-jp-eval-mm.git eval-mm
cd eval-mm
```

### 2. .env の作成

プロジェクトルートに `.env` を作成し、環境固有の設定を記述します：

```bash
# .env — TSUBAME 4.0 用設定
# 認証
HF_TOKEN=hf_xxx
OPENAI_API_KEY=sk-xxx

# パス
PROJECT_DIR=/gs/bs/<group>/<user>/eval-mm
RESULT_DIR=/gs/bs/<group>/<user>/eval-mm-results
LOG_DIR=/gs/bs/<group>/<user>/eval-mm-logs
HF_HOME=/gs/bs/<group>/<user>/cache/huggingface
UV_CACHE_DIR=/gs/bs/<group>/<user>/cache/uv
VLLM_CACHE_DIR=/gs/bs/<group>/<user>/cache/vllm

# PBS
TSUBAME_GROUP=tga-okazaki
TSUBAME_QUEUE=gpu_h100
TSUBAME_WALLTIME=24:00:00
```

`.env` は `.gitignore` に含まれているため、認証情報が GitHub に漏れる心配はありません。

### 3. uv のインストール（未導入の場合）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
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

ログは `$LOG_DIR` に出力されます。

## 結果の同期

TSUBAME の計算結果を mdx 環境に同期して、Web ダッシュボードで確認できます：

```bash
# mdx 側から実行
rsync -avz tsubame:/gs/bs/.../eval-mm-results/ ./result/
```

## カスタマイズ

### モデルリストの編集

`config.sh` 内の `MODEL_LIST` をコメントアウト/編集して、実行対象を調整してください。

### PBS リソースの変更

`run_model.sh` 先頭の `#PBS` ディレクティブを直接編集するか、`qsub` の引数で上書きしてください：

```bash
qsub -q other_queue -l walltime=12:00:00 -v MODEL_ENTRY="..." tsubame/run_model.sh
```

## 注意事項

- API ベースのモデル（gpt-4o 等）は計算ノードからインターネットアクセスできない場合があるため、デフォルトでコメントアウトしています
- 初回実行時はモデルのダウンロードに時間がかかります。事前にログインノードで `huggingface-cli download <model_id>` しておくことを推奨します
