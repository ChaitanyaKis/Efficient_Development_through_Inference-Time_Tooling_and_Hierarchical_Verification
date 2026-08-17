from dataclasses import dataclass
import collections

defaultdict = collections.defaultdict

@dataclass
class Task:
    title: str
    done: bool = False

class TaskStore:
    def __init__(self):
        self.tasks = defaultdict(list)

    def add(self, task: Task) -> None:
        self.tasks[task.title].append(task)

    def all(self) -> list[Task]:
        return [task for tasks in self.tasks.values() for task in tasks]
