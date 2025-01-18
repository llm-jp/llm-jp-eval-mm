# LLM-jp-eval-mm
[![pypi](https://img.shields.io/pypi/v/eval-mm.svg)](https://pypi.python.org/pypi/eval-mm)

This README is in Japanese and the latest information is in English. Please refer to the English version for the latest information.

[ [**English**](./README.md) | 日本語 ]

このツールは，複数のデータセットを横断して日本語マルチモーダル大規模言語モデルを自動評価するものです．
このツールは以下の機能を提供します：

- 既存の日本語評価データを利用し，マルチモーダルテキスト生成タスクの評価データセットに変換して提供する．
- ユーザが作成した推論結果を用いて，タスクごとに設定された評価メトリクスを計算する．

![llm-jp-eval-mmが提供するもの](https://github.com/llm-jp/llm-jp-eval-mm/blob/master/assets/teaser.png)

データフォーマットの詳細，サポートしているデータの一覧については，[DATASET.md](./DATASET.md)を参照ください．

## 目次

- [LLM-jp-eval-mm](#llm-jp-eval-mm)
  - [目次](#目次)
  - [環境構築](#環境構築)
    - [PyPIでインストールする](#pypiでインストールする)
    - [GitHubをCloneする場合](#githubをcloneする場合)
  - [評価方法](#評価方法)
    - [評価の実行](#評価の実行)
    - [リーダーボードの公開](#リーダーボードの公開)
  - [サポートするタスク](#サポートするタスク)
  - [各VLMモデル推論時の必要ライブラリ情報](#各vlmモデル推論時の必要ライブラリ情報)
  - [ベンチマーク固有の必要ライブラリ情報](#ベンチマーク固有の必要ライブラリ情報)
  - [ライセンス](#ライセンス)
  - [Contribution](#contribution)
    - [ベンチマークタスクの追加方法](#ベンチマークタスクの追加方法)
    - [メトリックの追加方法](#メトリックの追加方法)
    - [VLMモデルの推論コードの追加方法](#vlmモデルの推論コードの追加方法)
    - [依存ライブラリの追加方法](#依存ライブラリの追加方法)
    - [ruffを用いたフォーマット, リント](#ruffを用いたフォーマット-リント)
    - [PyPIへのリリース方法](#pypiへのリリース方法)
    - [Webサイトの更新方法](#webサイトの更新方法)
  - [Reference](#reference)

## 環境構築

このツールはPyPI経由で利用することができます．

### PyPIでインストールする

1. `pip`コマンドを用いて`eval_mm`を利用している仮想環境に含めることができます．

```bash
pip install eval_mm
```

2. 本ツールではLLM-as-a-judge手法を用いて評価をする際に，OpenAI APIを用いてGPT-4oにリクエストを送信します．`.env`ファイルを作成し，Azureを利用する場合には`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`の組を，OpenAI APIを利用する場合は`OPENAI_API_KEY`を設定してください.

以上で環境構築は終了です.

リポジトリをクローンして利用する場合は以下の手順を参考にしてください．

### GitHubをCloneする場合

eval-mmは仮想環境の管理にuvを用いています．

1. リポジトリをクローンして移動する
```bash
git clone git@github.com:llm-jp/llm-jp-eval-mm.git
cd llm-jp-eval-mm
```

2. uv を用いて環境構築を行う

uvは[official doc](https://docs.astral.sh/uv/getting-started/installation/) を参考にインストールしてください．

```bash
cd llm-jp-eval-mm
uv sync
```

3. [.env.sample](./.env.sample)を参考にして, `.env`ファイルを作成し，`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`の組，あるいは`OPENAI_API_KEY`を設定してください.

以上で環境構築は終了です.


## 評価方法

### 評価の実行

(現在, llm-jp-eval-mm リポジトリはprivateになっています. examples ディレクトリについては, [https://pypi.org/project/eval-mm/#files](https://pypi.org/project/eval-mm/#files)のSource Distributionにてdownloadできます.)

評価の実行のために，サンプルコード`examples/sample.py`を提供しています．

`examples/{モデル名}.py`として含まれているモデルは，その推論方法に限りサポートしています．

新たな推論方法・新たなモデルでの評価を実行したい場合，既存の`examples/{モデル名}.py`を参考に同様のファイルを作成することで，評価を実行することができます．

例として, `llava-hf/llava-1.5-7b-hf`モデルをjapanese-heron-benchで評価したい場合は, 以下のコマンドを実行してください．

```bash
python3 examples/sample.py \
  --model_id llava-hf/llava-1.5-7b-hf \
  --task_id japanese-heron-bench  \
  --result_dir test  \
  --metrics "llm_as_a_judge_heron_bench" \
  --judge_model "gpt-4o-2024-05-13" \
  --overwrite
```

評価結果のスコアと出力結果は
`test/{task_id}/evaluation/{model_id}.jsonl`, `test/{task_id}/prediction/{model_id}.jsonl` に保存されます.

複数のモデルを複数のタスクで評価したい場合は, `eval_all.sh`を参考にしてください.

### リーダーボードの公開

現在，代表的なモデルの評価結果をまとめたリーダーボードを公開する予定があります．

## サポートするタスク

現在，以下のベンチマークタスクをサポートしています．

Japanese Task:
- [Japanese Heron Bench](https://huggingface.co/datasets/turing-motors/Japanese-Heron-Bench)
- [JA-VG-VQA500](https://huggingface.co/datasets/SakanaAI/JA-VG-VQA-500)
- [JA-VLM-Bench-In-the-Wild](https://huggingface.co/datasets/SakanaAI/JA-VLM-Bench-In-the-Wild)
- [JA-Multi-Image-VQA](https://huggingface.co/datasets/SakanaAI/JA-Multi-Image-VQA)
- [JDocQA](https://huggingface.co/datasets/shunk031/JDocQA)
- [JMMMU](https://huggingface.co/datasets/JMMMU/JMMMU)

English Task:
- [MMMU](https://huggingface.co/datasets/MMMU/MMMU)
- [LlaVA-Bench-In-the-Wild](https://huggingface.co/datasets/lmms-lab/llava-bench-in-the-wild)

## 各VLMモデル推論時の必要ライブラリ情報

各モデルごとに, 必要なライブラリが異なります. このリポジトリでは, uvの[Dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups)を用いて, モデルごとに必要なライブラリを管理しています.

以下のモデルを利用する際には, normal groupを指定してください.
stabiliyai/japanese-instructblip-alpha, stabilityai/japanese-stable-vlm, cyberagent/llava-calm2-siglip, llava-hf/llava-1.5-7b-hf, llava-hf/llava-v1.6-mistral-7b-hf, neulab/Pangea-7B-hf,  meta-llama/Llama-3.2-11B-Vision-Instruct, meta-llama/Llama-3.2-90B-Vision-Instruct, OpenGVLab/InternVL2-8B, Qwen/Qwen2-VL-7B-Instruct, OpenGVLab/InternVL2-26B, Qwen/Qwen2-VL-72B-Instruct, gpt-4o-2024-05-13
```bash
uv sync --group normal
```

以下のモデルを利用する際には, evovlm groupを指定してください.
SamanaAI/Llama-3-EvoVLM-JP-v2

```bash
uv sync --group evovlm
```

以下のモデルを利用する際には, vilaja groupを指定してください.
llm-jp/llm-jp-3-vila-14b, Efficient-Large-Model/VILA1.5-13b
```bash
uv sync --group vilaja
```

mistralai/Pixtral-12B-2409
```bash
uv sync --group pixtral
```


実行時は, groupを指定してください.

```bash
$ uv run --group normal python ...
```

新しいgroupを追加する際は, [conflict](https://docs.astral.sh/uv/concepts/projects/config/#conflicting-dependencies)の設定を忘れないようにしてください.



## ベンチマーク固有の必要ライブラリ情報

- JDocQA
JDocQA データセットの構築において, [pdf2image](https://pypi.org/project/pdf2image/) library が必要です. pdf2imageはpoppler-utilsに依存していますので, 以下のコマンドでインストールしてください.
```bash
sudo apt-get install poppler-utils
```

## ライセンス

各評価データセットのライセンスは[DATASET.md](./DATASET.md)を参照してください．

## Contribution

- 問題や提案があれば，Issue で報告してください．
- 新たなベンチマークタスクやメトリック, VLMモデルの推論コードの追加や, バグの修正がありましたら, Pull Requestを送ってください.

### ベンチマークタスクの追加方法
タスクはTaskクラスで定義されます.
[src/eval_mm/tasks](https://github.com/llm-jp/llm-jp-eval-mm/blob/master/src/eval_mm/tasks)のコードを参考にTaskクラスを実装してください.
データセットをVLMモデルに入力する形式に変換するメソッドと, スコアを計算するメソッドを定義する必要があります.

### メトリックの追加方法
メトリックはScorerクラスで定義されます.
[src/eval_mm/metrics](https://github.com/llm-jp/llm-jp-eval-mm/blob/master/src/eval_mm/metrics)のコードを参考にScorerクラスを実装してください.
参照文と生成文を比較してsampleレベルのスコアリングを行う`score()`メソッドと, スコアを集約してpopulationレベルのメトリック計算を行う`aggregate()`メソッドを定義する必要があります.

### VLMモデルの推論コードの追加方法
VLMモデルの推論コードはVLMクラスで定義されます.
[examples/base_vlm](https://github.com/llm-jp/llm-jp-eval-mm/blob/master/examples/base_vlm.py)を参考に, VLMクラスを実装してください.
画像とプロンプトをもとに生成文を生成する`generate()`メソッドを定義する必要があります.


### 依存ライブラリの追加方法

```
uv add <package_name>
uv add --group <group_name> <package_name>
```

### ruffを用いたフォーマット, リント
```
uv run ruff format src
uv run ruff check --fix src
```

### PyPIへのリリース方法
```
git tag -a v0.x.x -m "version 0.x.x"
git push origin --tags
```

### Webサイトの更新方法
[github_pages/README.md](./github_pages/README.md)を参照ください.


## Reference
- https://github.com/EvolvingLMMs-Lab/lmms-eval
- https://github.com/turingmotors/heron
