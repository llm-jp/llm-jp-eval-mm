# TSUBAME 4.0 での実行方法

TSUBAME 4.0 (東工大 / 科学大) で llm-jp-eval-mm の評価を SGE ジョブとして実行するためのスクリプト群です。

## 前提

- Altair Grid Engine (SGE互換) によるジョブスケジューリング
- H100 SXM5 x 4 / ノード (node_f)
- Python 環境: uv

## ファイル構成

```
tsubame/
├── config.sh       # 共通定義（モデルリスト、タスク、メトリクス）
├── run_model.sh    # SGE ジョブスクリプト（1モデル = 1ジョブ）
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

```bash
cat > .env << 'EOF'
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

# SGE
TSUBAME_GROUP=tga-okazaki
TSUBAME_RESOURCE=node_f
TSUBAME_H_RT=24:00:00
EOF
```

### リソースタイプ

| 指定 | GPU | CPU | メモリ |
|------|-----|-----|--------|
| `node_f` | 4 | 192 | 768GB |
| `node_h` | 2 | 96 | 384GB |
| `node_q` | 1 | 48 | 192GB |

## 使い方

### dry-run で確認

```bash
bash tsubame/submit_all.sh --dry-run
```

### 全モデルを一括投入

```bash
bash tsubame/submit_all.sh
```

### フィルタリング

```bash
bash tsubame/submit_all.sh --filter vllm           # vLLM モデルのみ
bash tsubame/submit_all.sh --filter transformers    # transformers モデルのみ
bash tsubame/submit_all.sh --grep "Qwen"           # Qwen 系のみ
```

### 単一モデルを手動投入

```bash
qsub -g tga-okazaki -l node_f=1 -l h_rt=24:00:00 \
  -v MODEL_ENTRY="Qwen/Qwen2.5-VL-7B-Instruct|vllm_normal|vllm|1" \
  tsubame/run_model.sh
```

## ジョブの確認

```bash
qstat                       # 自分のジョブ一覧
qstat -j <job_id>           # ジョブ詳細
qdel <job_id>               # ジョブ削除
```

## 結果の同期

TSUBAME の計算結果を mdx 環境に同期して、Web ダッシュボードで確認できます：

```bash
# mdx 側から実行
rsync -avz tsubame:/gs/bs/.../eval-mm-results/ ./result/
```

## 注意事項

- API ベースのモデル（gpt-4o 等）は計算ノードからインターネットアクセスできない場合があるため、デフォルトでコメントアウトしています
- 初回実行時はモデルのダウンロードに時間がかかります。事前にログインノードで `huggingface-cli download <model_id>` しておくことを推奨します
- `-g` を指定しないとお試し実行扱い（最大3分、ノード2台まで）になります
