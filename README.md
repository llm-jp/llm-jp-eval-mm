# llm-jp-eval-mm
[![pypi](https://img.shields.io/pypi/v/eval-mm.svg)](https://pypi.python.org/pypi/eval-mm) [![Test workflow](https://github.com/llm-jp/llm-jp-eval-mm/actions/workflows/test.yml/badge.svg)](https://github.com/llm-jp/llm-jp-eval-mm/actions/workflows/test.yml) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

llm-jp-eval-mm is a lightweight framework for evaluating visual-language models across various benchmark tasks, mainly focusing on Japanese tasks.

![Overview of llm-jp-eval-mm](https://github.com/llm-jp/llm-jp-eval-mm/blob/master/assets/teaser.png)

## Getting Started

You can install llm-jp-eval-mm from GitHub or via PyPI.

- Option 1: Clone from GitHub (Recommended)
```bash
git clone git@github.com:llm-jp/llm-jp-eval-mm.git
cd llm-jp-eval-mm
uv sync
```

- Option 2: Install via PyPI
```bash
pip install eval_mm
```

To use LLM-as-a-Judge, configure your OpenAI API keys in a`.env` file:
- For Azure: Set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_KEY`
- For OpenAI: Set `OPENAI_API_KEY`

If you are not using LLM-as-a-Judge, you can assign any value in the `.env` file to bypass the error.

## Usage

To evaluate a model on a task, run the following command:
```bash
uv sync --group normal
uv run --group normal python examples/sample.py \
  --model_id llava-hf/llava-1.5-7b-hf \
  --task_id japanese-heron-bench  \
  --result_dir result  \
  --metrics heron-bench \
  --judge_model gpt-4o-2024-11-20 \
  --overwrite
```

The evaluation results will be saved in the result directory:
```
result
├── japanese-heron-bench
│   ├── llava-hf
│   │   ├── llava-1.5-7b-hf
│   │   │   ├── evaluation.jsonl
│   │   │   └── prediction.jsonl
```

To evaluate multiple models on multiple tasks, please check `eval_all.sh`.

## Hello World Example

You can integrate llm-jp-eval-mm into your own code. Here's an example:
```python
from PIL import Image
from eval_mm import TaskRegistry, ScorerRegistry, ScorerConfig

class MockVLM:
    def generate(self, images: list[Image.Image], text: str) -> str:
        return "宮崎駿"

task = TaskRegistry.load_task("japanese-heron-bench")
example = task.dataset[0]

input_text = task.doc_to_text(example)
images = task.doc_to_visual(example)
reference = task.doc_to_answer(example)

model = MockVLM()
prediction = model.generate(images, input_text)

scorer = ScorerRegistry.load_scorer(
    "rougel",
    ScorerConfig(docs=task.dataset)
)
result = scorer.aggregate(scorer.score([reference], [prediction]))
print(result)
# AggregateOutput(overall_score=5.128205128205128, details={'rougel': 5.128205128205128})
```


## Leaderboard

To generate a leaderboard from your evaluation results, run:
```bash
python scripts/make_leaderboard.py --result_dir result
```

This will create a `leaderboard.md` file with your model performance:

| Model                                    | Heron/LLM | JVB-ItW/LLM | JVB-ItW/Rouge |
| :--------------------------------------- | :-------- | :---------- | :------------ |
| llm-jp/llm-jp-3-vila-14b                 | 68.03     | 4.08        | **52.4**      |
| Qwen/Qwen2.5-VL-7B-Instruct              | 70.29     | 4.28        | 29.63         |
| google/gemma-3-27b-it                    | 69.15     | 4.36        | 30.89         |
| microsoft/Phi-4-multimodal-instruct      | 45.52     | 3.2         | 26.8          |
| gpt-4o-2024-11-20                        | **93.7**  | **4.44**    | 32.2          |



The official leaderboard is available [here](https://llm-jp.github.io/llm-jp-eval-mm/)

## Supported Tasks

Japanese Tasks:
- [Japanese Heron Bench](https://huggingface.co/datasets/turing-motors/Japanese-Heron-Bench)
- [JA-VG-VQA500](https://huggingface.co/datasets/SakanaAI/JA-VG-VQA-500)
- [JA-VLM-Bench-In-the-Wild](https://huggingface.co/datasets/SakanaAI/JA-VLM-Bench-In-the-Wild)
- [JA-Multi-Image-VQA](https://huggingface.co/datasets/SakanaAI/JA-Multi-Image-VQA)
- [JDocQA](https://github.com/mizuumi/JDocQA)
- [JMMMU](https://huggingface.co/datasets/JMMMU/JMMMU)
- [JIC-VQA](https://huggingface.co/datasets/line-corporation/JIC-VQA)
- [MECHA-ja](https://huggingface.co/datasets/llm-jp/MECHA-ja)
- [CC-OCR](https://huggingface.co/datasets/wulipc/CC-OCR) (multi_lan_ocr split, ja subset)
- [CVQA](https://huggingface.co/datasets/afaji/cvqa) (ja subset)

English Tasks:
- [MMMU](https://huggingface.co/datasets/MMMU/MMMU)
- [LlaVA-Bench-In-the-Wild](https://huggingface.co/datasets/lmms-lab/llava-bench-in-the-wild)

## Managing Dependencies

We use uv’s dependency groups to manage each model’s dependencies.

For example, to use llm-jp/llm-jp-3-vila-14b, run:
```bash
uv sync --group vilaja
uv run --group vilaja python examples/VILA_ja.py
```

See `eval_all.sh` for the complete list of model dependencies.

When adding a new group, remember to configure [conflict](https://docs.astral.sh/uv/concepts/projects/config/#conflicting-dependencies).

## Browse Predictions with Streamlit
```bash
uv run streamlit run scripts/browse_prediction.py -- --task_id japanese-heron-bench --result_dir result --model_list llava-hf/llava-1.5-7b-hf
```

![Streamlit](./assets/streamlit_visualization.png)


## Development

### Adding a new task

To add a new task, implement the Task class in `src/eval_mm/tasks/task.py`.

### Adding a new metric

To add a new metric, implement the Scorer class in `src/eval_mm/metrics/scorer.py`.

### Adding a new model

To add a new model, implement the VLM class in `examples/base_vlm.py`

### Adding a new dependency

Install a new dependency using the following command:
```bash
uv add <package_name>
uv add --group <group_name> <package_name>
```


### Testing

Run the following commands to test tasks, metrics, and models::
```bash
bash test.sh
bash test_model.sh
```

### Formatting and Linting

Ensure code consistency with:
```bash
uv run ruff format src
uv run ruff check --fix src
```

### Releasing to PyPI

To release a new version:
```bash
git tag -a v0.x.x -m "version 0.x.x"
git push origin --tags
```


### Updating the Website

For website updates, see [github_pages/README.md](./github_pages/README.md).

To update leaderboard data:
```bash
python scripts/make_leaderboard.py --update_pages
```

## Acknowledgements
- [Heron](https://github.com/turingmotors/heron): We refer to the Heron code for the evaluation of the Japanese Heron Bench task.
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): We refer to the lmms-eval code for the evaluation of the JMMMU and MMMU tasks.

We also thank the developers of the evaluation datasets for their hard work.
