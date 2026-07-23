from dataclasses import dataclass
from pathlib import Path
import shutil


PROJECT_MEMORY_SKILL = (
    Path(__file__).resolve().parents[2] / "skills" / "memorizer-memory"
)


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path

    @property
    def trajectory(self) -> Path:
        return self.root / "data" / "trajectory" / "trajectory.jsonl"

    @property
    def memories(self) -> Path:
        return self.root / "memory" / "Memories.md"

    @property
    def ledger(self) -> Path:
        return self.root / "memory" / "ledger" / "memory_ledger.jsonl"

    @property
    def snapshots(self) -> Path:
        return self.root / "memory" / "snapshots"

    @property
    def diffs(self) -> Path:
        return self.root / "memory" / "diffs"

    @property
    def staging(self) -> Path:
        return self.root / "memory" / "staging"

    @property
    def archive(self) -> Path:
        return self.root / "memory" / "staging" / "archive"

    @property
    def runtime(self) -> Path:
        return self.root / "runtime"

    @property
    def consumption(self) -> Path:
        return self.runtime / "reasoner_consumption.json"

    @property
    def logs(self) -> Path:
        return self.runtime / "logs"

    @property
    def reasoner_log(self) -> Path:
        return self.logs / "reasoner.jsonl"

    @property
    def memorizer_log(self) -> Path:
        return self.logs / "memorizer.jsonl"

    @property
    def skill_log(self) -> Path:
        return self.logs / "skill.jsonl"

    @property
    def token_usage_log(self) -> Path:
        return self.logs / "token_usage.jsonl"

    @property
    def memory_skill(self) -> Path:
        return self.root / "skills" / "memorizer-memory"

    def ensure(self) -> None:
        self.trajectory.parent.mkdir(parents=True, exist_ok=True)
        self.memories.parent.mkdir(parents=True, exist_ok=True)
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.diffs.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)
        self.archive.mkdir(parents=True, exist_ok=True)
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.consumption.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_memory_skill()

    def _ensure_memory_skill(self) -> None:
        if not PROJECT_MEMORY_SKILL.exists():
            raise FileNotFoundError(
                f"Canonical memory skill not found: {PROJECT_MEMORY_SKILL}"
            )
        if self.memory_skill.resolve() == PROJECT_MEMORY_SKILL.resolve():
            return
        shutil.copytree(PROJECT_MEMORY_SKILL, self.memory_skill, dirs_exist_ok=True)
