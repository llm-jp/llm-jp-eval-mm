"""
Task registry with decorator-based registration to avoid duplication.
"""

from typing import Type, Callable
from .task import Task, TaskConfig

# Global registry dictionary
_task_registry: dict[str, Type[Task]] = {}


def register_task(*names: str):
    """
    Decorator to register a task class in the global registry.
    Can register multiple names for the same task.
    
    Usage:
        @register_task("my-task-name", "MyTaskName", "MY_TASK_NAME")
        class MyTask(Task):
            ...
    """
    def decorator(cls: Type[Task]) -> Type[Task]:
        for name in names:
            _task_registry[name] = cls
        return cls
    return decorator


class TaskRegistry:
    """Registry to map metrics to their corresponding scorer classes."""

    @classmethod
    def get_task_list(cls) -> list[str]:
        """Get list of all registered task names."""
        return list(_task_registry.keys())

    @classmethod
    def load_task(cls, task_name: str, task_config: TaskConfig = TaskConfig()) -> Task:
        """Load a task by name."""
        try:
            return _task_registry[task_name](task_config)
        except KeyError:
            raise ValueError(f"Task '{task_name}' is not supported.")