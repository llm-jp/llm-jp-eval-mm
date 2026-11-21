"""Validate that metadata in eval_mm.metadata is consistent with registries."""


from eval_mm.metadata import TASKS, METRICS
from eval_mm.tasks.task_registry import TaskRegistry
from eval_mm.metrics.scorer_registry import ScorerRegistry


# Tasks that are registered but intentionally excluded from metadata
# (e.g., test-only tasks not shown on leaderboard)
_UNLISTED_TASKS = {"mnist"}


def test_all_registered_tasks_have_metadata():
    registered = set(TaskRegistry.get_task_list())
    metadata_ids = set(TASKS.keys())
    missing = registered - metadata_ids - _UNLISTED_TASKS
    assert not missing, f"Registered tasks missing from metadata: {missing}"


def test_all_metadata_tasks_are_registered():
    registered = set(TaskRegistry.get_task_list())
    metadata_ids = set(TASKS.keys())
    extra = metadata_ids - registered
    assert not extra, f"Metadata tasks not registered: {extra}"


def test_all_default_metrics_exist():
    registered_metrics = set(ScorerRegistry.get_metric_list())
    for task in TASKS.values():
        for metric_id in task.default_metrics:
            assert metric_id in registered_metrics, (
                f"Task '{task.task_id}' has default metric '{metric_id}' "
                f"which is not registered"
            )


def test_display_names_are_unique():
    names = [t.display_name for t in TASKS.values()]
    dupes = [n for n in names if names.count(n) > 1]
    assert not dupes, f"Duplicate display names: {set(dupes)}"


def test_metric_alias_covers_default_metrics():
    for task in TASKS.values():
        for metric_id in task.default_metrics:
            assert metric_id in METRICS, (
                f"Task '{task.task_id}' default metric '{metric_id}' "
                f"has no entry in METRICS"
            )
