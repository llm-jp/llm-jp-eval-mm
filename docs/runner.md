# Runner: リアルタイム評価モニタリング

## アーキテクチャ

```
┌─────────────────┐     polling (5s)    ┌──────────────────┐
│  Next.js (3000)  │ ◄────────────────► │  FastAPI (8000)   │
│  Runner Dashboard │                    │  eval_mm.api      │
└─────────────────┘                     └────────┬─────────┘
                                                 │ reads
                                        ┌────────▼─────────┐
                                        │  result/          │
                                        │  ├─ .eval_status  │  ◄── eval.sh writes
                                        │  ├─ eval_failures │
                                        │  └─ <task>/<model>│
                                        └──────────────────┘
```

**3つのプロセスが独立して動作する:**

| プロセス | 役割 | ポート |
|----------|------|--------|
| Next.js dev server | ダッシュボードUI | 3000 |
| FastAPI (uvicorn) | GPU情報・評価ステータスAPI | 8000 |
| eval.sh | モデル評価の実行 | - |

## 仮想環境の分離戦略

### 問題

`eval.sh` はモデルごとに `uv sync --group <group>` を実行し、`.venv` 内のパッケージ（torch, transformers, vllm等）を入れ替える。APIサーバーが同じ `.venv` に依存していると不安定になる。

### 解決策

```
.venv/
├── fastapi, uvicorn     ← base deps（conflict groupに含まれない → 常に存在）
├── torch, transformers   ← model group deps（uv sync で入れ替わる）
└── vllm                  ← model group deps（uv sync で入れ替わる）
```

1. **APIサーバーは `.venv/bin/python` で直接起動**（`uv run` ではない）
2. 起動時にPythonプロセスがモジュールをメモリにロード
3. その後 `uv sync` がディスク上のファイルを変更しても、**実行中のプロセスには影響しない**
4. fastapi / uvicorn は base dependencies であり、どの `uv sync --group` でも削除されない

## クイックスタート

### 1. モニタリングスタックを起動

```bash
bash scripts/start_runner.sh
```

これで API (8000) と Web (3000) が起動する。

### 2. 評価を実行

別ターミナルで:

```bash
# 全モデル実行
bash eval.sh

# vLLM バックエンドのモデルのみ実行
EVAL_BACKEND_FILTER=vllm bash eval.sh

# transformers バックエンドのモデルのみ実行
EVAL_BACKEND_FILTER=transformers bash eval.sh
```

### 3. ダッシュボードを確認

ブラウザで http://localhost:3000/runner を開く。

## API エンドポイント

| エンドポイント | 用途 | ポーリング間隔 |
|---------------|------|---------------|
| `GET /api/gpus` | GPU使用率・温度・メモリ (nvidia-smi) | 5秒 |
| `GET /api/run/status` | 現在の評価進捗 (.eval_status.json) | 5秒 |
| `GET /api/run/results` | タスク×モデルのpass/fail/runningマトリクス | 10秒 |
| `GET /api/tasks` | タスク一覧 (metadata) | 初回のみ |
| `GET /api/models` | モデル一覧 (metadata) | 初回のみ |
| `GET /api/results` | 結果ディレクトリの一覧 | オンデマンド |
| `GET /api/scores/{task}` | タスク別の集計スコア | オンデマンド |

## eval.sh のステータスファイル

`eval.sh` は各モデル/タスク実行前に `result/.eval_status.json` を更新する:

```json
{
  "running": true,
  "currentTask": "jmmmu",
  "currentModel": "Qwen/Qwen2-VL-7B-Instruct",
  "backend": "vllm",
  "completed": 42,
  "failed": 1,
  "total": 1600,
  "progress": 2,
  "etaSeconds": 456000,
  "elapsedSeconds": 3600
}
```

## ポートのカスタマイズ

```bash
API_PORT=9000 WEB_PORT=4000 bash scripts/start_runner.sh
```

Web側の環境変数 `NEXT_PUBLIC_API_URL` は自動的に設定される。

## トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| GPU データが表示されない | nvidia-smi が使えない | GPUマシンで実行しているか確認 |
| Results マトリクスが空 | eval.sh 未実行 | eval.sh を起動する |
| API に接続できない | ポート衝突 | `API_PORT` を変更 |
| eval.sh 中にAPIが落ちる | `uv run` で起動した | `scripts/start_runner.sh` を使う |
