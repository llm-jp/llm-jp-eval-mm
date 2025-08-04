# CLAUDE.md - llm-jp-eval-mm Development Guide

This document provides guidance for working with the llm-jp-eval-mm repository, including implementation tips and common workflows.

## Repository Overview

llm-jp-eval-mm is a multimodal evaluation framework for Vision-Language Models (VLMs) with support for Japanese and English tasks.

### Key Components

- **Tasks** (`src/eval_mm/tasks/`): Define evaluation datasets and how to load them
- **Metrics/Scorers** (`src/eval_mm/metrics/`): Define how to evaluate model outputs
- **Models** (`src/eval_mm/models/`): Model-specific implementations
- **Scripts** (`scripts/`): Utilities for running evaluations and generating leaderboards

## Adding a New Task

When implementing a new task, you need to update the following files:

### 1. Create Task Implementation

Create `src/eval_mm/tasks/<task_name>.py`:

```python
from eval_mm.tasks.task import Task
from eval_mm.tasks.task_registry import register_task
from datasets import load_dataset, Dataset
from PIL import Image

@register_task("task-id", "TaskName", "alternative-name")
class TaskName(Task):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def _prepare_dataset() -> Dataset:
        # Load and prepare dataset
        pass

    @staticmethod
    def doc_to_text(doc) -> str:
        # Convert to text prompt
        pass

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        # Extract images
        pass

    @staticmethod
    def doc_to_id(doc) -> str:
        # Return unique ID
        pass

    @staticmethod
    def doc_to_answer(doc) -> str | list[str]:
        # Return answer(s)
        pass
```

### 2. Update Task Registry

Edit `src/eval_mm/tasks/__init__.py`:

- Add import: `from .<task_name> import TaskName`
- Add to `__all__` list: `"TaskName"`

### 3. Update Configuration Files

#### For HPC/NVLink usage (`scripts/nvlink/config.sh`)

- Add to `task_list` array
- Add to `METRIC_MAP` with appropriate scorer

#### For leaderboard generation (`scripts/make_leaderboard.py`)

- Add to `TASK_ALIAS` dictionary
- Add to `TASK_CLUSTER_ALIAS` (categorize as 言語・知識中心, 視覚中心, その他, or 英語)
- Add to `METRIC_ALIAS` dictionary

### 4. Create/Select Appropriate Scorer

For multiple-choice tasks:

- Use existing scorers like `AI2DScorer` or `BLINKScorer`
- Implement custom normalization if needed

For extractive/generation tasks:

- Use `substring-match` for answers with variations
- Use `llm-as-a-judge` for open-ended generation
- Use `exact-match` for strict matching

## Common Scorer Types

1. **Multiple Choice**: Normalize answers (remove spaces, punctuation, case)
2. **Substring Match**: Check if any valid answer appears in prediction
3. **LLM as Judge**: Use GPT-4 to evaluate quality
4. **Exact Match**: Strict string comparison
5. **Rouge-L**: For summarization tasks

## Task Implementation Examples

### Multiple Choice Task (e.g., BLINK, AI2D)

- Provides choices and correct answer letter
- Uses custom scorer with answer normalization

### Extractive QA Task (e.g., DocVQA)

- No predefined choices
- Multiple valid answers possible
- Uses substring-match scorer

### Generation Task (e.g., JDocQA)

- Open-ended questions
- Uses LLM-as-a-judge for evaluation

## Running Evaluations

### Single model, single task

```bash
bash scripts/nvlink/sbatch.sh "model-name" 1 --tasks task-name
```

### All models, single task

```bash
bash scripts/nvlink/sbatch.sh --all --tasks task-name
```

### Multiple tasks

```bash
bash scripts/nvlink/sbatch.sh "model-name" 1 --tasks task1,task2,task3
```

## Import Patterns

Always use `import eval_mm` pattern (not `from src.eval_mm`):

```python
import eval_mm
task = eval_mm.TaskRegistry.load_task('task-name')
scorer = eval_mm.ScorerRegistry.load_scorer('scorer-name')
```

## Testing Your Implementation

Include a test function in your task file:

```python
def test_task_name():
    from eval_mm.tasks.task import TaskConfig
    task = TaskName(TaskConfig(max_dataset_len=10))
    # Test all methods
    # Verify data types
```

Run with:

```bash
source .uv/dev-env/bin/activate
python -c "from src.eval_mm.tasks.task_name import test_task_name; test_task_name()"
```

## Best Practices

1. **Dataset Loading**: Use HuggingFace datasets when possible
2. **Answer Format**: Be consistent with answer formats across similar tasks
3. **Error Handling**: Handle missing images or fields gracefully
4. **Documentation**: Include docstrings explaining task specifics
5. **Normalization**: Document any answer normalization in scorer

## Common Issues

1. **Import not found**: Ensure you've added imports to `__init__.py`
2. **Task not registered**: Check decorator syntax and import order
3. **Scorer mismatch**: Verify metric mapping in config files
4. **GPU allocation**: Check model_gpu_map in config.sh

## Environment Setup

The repository uses two virtual environments:

- `.venv/`: Standard Python environment
- `.uv/dev-env/`: UV-managed environment (preferred)

Always activate the appropriate environment:

```bash
source .uv/dev-env/bin/activate
```

## Commit Guidelines

When adding a new task:

1. Stage only relevant files
2. Use clear commit messages
3. Include co-author attribution for Claude-assisted code
4. Test thoroughly before committing

Example commit:

```sh
feat: Add DocVQA task implementation

- Add DocVQA task for document visual QA
- Use substring-match scorer for answer variations
- Update task registry and configuration files

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
```
