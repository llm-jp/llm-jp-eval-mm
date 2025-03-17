# llm-jp-eval-mm
[![pypi](https://img.shields.io/pypi/v/eval-mm.svg)](https://pypi.python.org/pypi/eval-mm)

[ [**Japanese**](./README_ja.md) | English ]

This tool automatically evaluates Japanese multi-modal large language models across multiple datasets. It offers the following features:

- Uses existing Japanese evaluation data and converts it into multi-modal text generation tasks for evaluation.
- Calculates task-specific evaluation metrics using inference results created by users.

![What llm-jp-eval-mm provides](https://github.com/llm-jp/llm-jp-eval-mm/blob/master/assets/teaser.png)

For details on the data format and the list of supported data, please check [DATASET.md](./DATASET.md).

## Table of Contents

- [llm-jp-eval-mm](#llm-jp-eval-mm)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
  - [How to Evaluate](#how-to-evaluate)
    - [Running an Evaluation](#running-an-evaluation)
    - [Leaderboard](#leaderboard)
  - [Supported Tasks](#supported-tasks)
  - [Required Libraries for Each VLM Model Inference](#required-libraries-for-each-vlm-model-inference)
  - [Benchmark-Specific Required Libraries](#benchmark-specific-required-libraries)
  - [License](#license)
  - [Contribution](#contribution)
    - [How to Add a Benchmark Task](#how-to-add-a-benchmark-task)
    - [How to Add a Metric](#how-to-add-a-metric)
    - [How to Add Inference Code for a VLM Model](#how-to-add-inference-code-for-a-vlm-model)
    - [How to Add Dependencies](#how-to-add-dependencies)
    - [Formatting and Linting with ruff](#formatting-and-linting-with-ruff)
    - [Testing](#testing)
    - [How to Release to PyPI](#how-to-release-to-pypi)
    - [How to Update the Website](#how-to-update-the-website)
  - [Acknowledgements](#acknowledgements)

## Getting Started

You can use this tool via GitHub (Recommended).

```bash
git clone git@github.com:llm-jp/llm-jp-eval-mm.git
cd llm-jp-eval-mm
uv sync
```

Or you can install it via PyPI.
```bash
pip install eval_mm
```

This tool uses the LLM-as-a-judge method for evaluation, which sends requests to GPT-4o via the OpenAI API. Please create a `.env` file and set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_KEY` if you’re using Azure, or `OPENAI_API_KEY` if you’re using the OpenAI API.

That’s it! You’re ready to evaluate your VLM model.

## How to Evaluate

### Running an Evaluation

We provide a sample code `examples/sample.py` for running an evaluation.

Models listed as `examples/{model_name}.py` are supported only in terms of their inference method.

If you want to run an evaluation on a new inference method or a new model, create a similar file referencing existing `examples/{model_name}.py`, and you can run the evaluation in the same way.

For example, if you want to evaluate the `llava-hf/llava-1.5-7b-hf` model on japanese-heron-bench task, run the following command:

```bash
uv sync --group normal
uv run --group normal python examples/sample.py \
  --model_id llava-hf/llava-1.5-7b-hf \
  --task_id japanese-heron-bench  \
  --result_dir test  \
  --metrics "llm_as_a_judge_heron_bench" \
  --judge_model "gpt-4o-2024-05-13" \
  --overwrite
```

The evaluation score and output results will be saved in
`test/{task_id}/{model_id}/evaluation.jsonl` and `test/{task_id}/{model_id}/prediction/.jsonl`.

If you want to evaluate multiple models on multiple tasks, please check `eval_all.sh`.

### Leaderboard

Leaderboard is [here](https://llm-jp.github.io/llm-jp-eval-mm/)

## Supported Tasks

Right now, the following benchmark tasks are supported:

Japanese Task:
- [Japanese Heron Bench](https://huggingface.co/datasets/turing-motors/Japanese-Heron-Bench)
- [JA-VG-VQA500](https://huggingface.co/datasets/SakanaAI/JA-VG-VQA-500)
- [JA-VLM-Bench-In-the-Wild](https://huggingface.co/datasets/SakanaAI/JA-VLM-Bench-In-the-Wild)
- [JA-Multi-Image-VQA](https://huggingface.co/datasets/SakanaAI/JA-Multi-Image-VQA)
- [JDocQA](https://github.com/mizuumi/JDocQA)
- [JMMMU](https://huggingface.co/datasets/JMMMU/JMMMU)
- [JIC-VQA](https://huggingface.co/datasets/line-corporation/JIC-VQA)

English Task:
- [MMMU](https://huggingface.co/datasets/MMMU/MMMU)
- [LlaVA-Bench-In-the-Wild](https://huggingface.co/datasets/lmms-lab/llava-bench-in-the-wild)

## Required Libraries for Each VLM Model Inference

Different models require different libraries. In this repository, we use uv’s [Dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups) to manage the libraries needed for each model.

When using the following models, please specify the `normal` group:
stabiliyai/japanese-instructblip-alpha, stabilityai/japanese-stable-vlm, cyberagent/llava-calm2-siglip, llava-hf/llava-1.5-7b-hf, llava-hf/llava-v1.6-mistral-7b-hf, neulab/Pangea-7B-hf, meta-llama/Llama-3.2-11B-Vision-Instruct, meta-llama/Llama-3.2-90B-Vision-Instruct, OpenGVLab/InternVL2-8B, Qwen/Qwen2-VL-7B-Instruct, OpenGVLab/InternVL2-26B, Qwen/Qwen2-VL-72B-Instruct, gpt-4o-2024-05-13
```bash
uv sync --group normal
```

When using the following model, please specify the `evovlm` group:
SamanaAI/Llama-3-EvoVLM-JP-v2
```bash
uv sync --group evovlm
```

When using the following models, please specify the `vilaja` group:
llm-jp/llm-jp-3-vila-14b, Efficient-Large-Model/VILA1.5-13b
```bash
uv sync --group vilaja
```

mistralai/Pixtral-12B-2409
```bash
uv sync --group pixtral
```

When running the script, make sure to specify the group:

```bash
$ uv run --group normal python ...
```

If you add a new group, don’t forget to configure [conflict](https://docs.astral.sh/uv/concepts/projects/config/#conflicting-dependencies).

## Benchmark-Specific Required Libraries

- JDocQA

To prepare the JDocQA dataset, [pdf2image](https://pypi.org/project/pdf2image/) library is needed. Since pdf2image depends on poppler-utils, please install it with:

```bash
sudo apt-get install poppler-utils
```

- JIC-VQA

JIC-VQA only provide the image URL, so you need to download the images from the URL. You can use the following code to prepare the JIC-VQA dataset with the image download.

```python
python scripts/prepare_jic_vqa.py
```

## License

This repository is licensed under the Apache-2.0 License.

## Contribution

- If you find any issues or have suggestions, please report them on the Issue.
- If you add new benchmark tasks, metrics, or VLM model inference code, or if you fix bugs, please send us a Pull Request.

### How to Add a Benchmark Task
Tasks are defined in the `Task` class.
Please reference the code in [src/eval_mm/tasks](https://github.com/llm-jp/llm-jp-eval-mm/blob/master/src/eval_mm/tasks) and implement your `Task` class. You’ll need methods to convert the dataset into a format for input to the VLM model, and methods to calculate the score.

### How to Add a Metric
Metrics are defined in the `Scorer` class.
Please reference the code in [src/eval_mm/metrics](https://github.com/llm-jp/llm-jp-eval-mm/blob/master/src/eval_mm/metrics) and implement your `Scorer` class. You’ll need to implement a `score()` method for sample-level scoring comparing references and generated outputs, and an `aggregate()` method for population-level metric calculation.

### How to Add Inference Code for a VLM Model
Inference code for VLM models is defined in the `VLM` class.
Please reference [examples/base_vlm](https://github.com/llm-jp/llm-jp-eval-mm/blob/master/examples/base_vlm.py) and implement your `VLM` class. You’ll need a `generate()` method to output text given images and text inputs.

### How to Add Dependencies

```
uv add <package_name>
uv add --group <group_name> <package_name>
```

### Formatting and Linting with ruff
```
uv run ruff format src
uv run ruff check --fix src
```

### Testing

You can test task classes and metric classes with the following command:
```bash
bash test.sh
```
You can also test each model's inference code with the following command:
```bash
bash test_model.sh
```


### How to Release to PyPI

```
git tag -a v0.x.x -m "version 0.x.x"
git push origin --tags
```
Or you can manually create a new release on GitHub.


### How to Update the Website
Please refer to [github_pages/README.md](./github_pages/README.md).

## Acknowledgements
- [Heron](https://github.com/turingmotors/heron): We refer to the Heron code for the evaluation of the Japanese Heron Bench task.
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): We refer to the lmms-eval code for the evaluation of the JMMMU and MMMU tasks.

We also thank the developers of the evaluation datasets for their hard work.
