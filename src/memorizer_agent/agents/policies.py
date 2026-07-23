from __future__ import annotations

from pathlib import Path
from typing import Iterable

from memorizer_agent.paths import WorkspacePaths


class HarnessViolation(RuntimeError):
    """Raised when an agent attempts an action outside its harness boundary."""


class ReasonerPolicy:
    def __init__(self, paths: WorkspacePaths) -> None:
        self.paths = paths
        self._readable = {
            paths.memories.resolve(),
            paths.trajectory.resolve(),
        }

    def check_read(self, path: Path) -> None:
        if path.resolve() not in self._readable:
            raise HarnessViolation(f"Reasoner cannot read {path}")

    def check_write(self, path: Path) -> None:
        raise HarnessViolation(f"Reasoner cannot write {path}")

    def check_command(self, command: Iterable[str]) -> None:
        tokens = [str(token) for token in command]
        if _contains_memory_skill_command(tokens) or _contains_app_memory_command(tokens):
            raise HarnessViolation("Reasoner cannot run memory maintenance commands")


class MemorizerPolicy:
    def __init__(self, paths: WorkspacePaths) -> None:
        self.paths = paths
        self._protected_roots = [
            paths.memories.resolve(),
            paths.ledger.resolve(),
            paths.snapshots.resolve(),
            paths.diffs.resolve(),
            paths.staging.resolve(),
        ]
        self._skill_memory_script = (
            Path("skills") / "memorizer-memory" / "scripts" / "memory.py"
        ).as_posix()

    def check_read(self, path: Path) -> None:
        path.resolve()

    def check_write(self, path: Path) -> None:
        resolved = path.resolve()
        for protected in self._protected_roots:
            if resolved == protected or _is_relative_to(resolved, protected):
                raise HarnessViolation(f"Memorizer cannot directly write durable memory path {path}")

    def check_command(self, command: Iterable[str]) -> None:
        tokens = [str(token) for token in command]
        if _contains_app_memory_command(tokens):
            raise HarnessViolation("Memorizer cannot call app-level memory commands")
        if _contains_memory_py_name(tokens) and not _contains_memory_skill_command(tokens):
            raise HarnessViolation("Memorizer must call the skill-local memory.py")


def _contains_app_memory_command(tokens: list[str]) -> bool:
    normalized = [token.lower() for token in tokens]
    return "memorizer-agent" in normalized and "memory" in normalized


def _contains_memory_py_name(tokens: list[str]) -> bool:
    return any(Path(token).name == "memory.py" for token in tokens)


def _contains_memory_skill_command(tokens: list[str]) -> bool:
    expected = (Path("skills") / "memorizer-memory" / "scripts" / "memory.py").as_posix()
    for token in tokens:
        normalized = token.replace("\\", "/")
        if normalized == expected or normalized.endswith(f"/{expected}"):
            return True
    return False


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
