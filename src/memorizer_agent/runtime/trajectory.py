import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from memorizer_agent.paths import WorkspacePaths


@dataclass(frozen=True)
class TrajectoryRecord:
    line_offset: int
    record_id: str


class TrajectoryRecorder:
    def __init__(self, paths: WorkspacePaths) -> None:
        self.paths = paths

    def record(
        self,
        speaker: str,
        content: str,
        *,
        time: str | datetime | None = None,
        record_id: str | None = None,
    ) -> TrajectoryRecord:
        self.paths.trajectory.parent.mkdir(parents=True, exist_ok=True)
        line_offset = self._next_line_offset()
        resolved_id = record_id or str(uuid.uuid4())
        event = {
            "id": resolved_id,
            "speaker": speaker,
            "time": _format_time(time),
            "content": content.encode("utf-8").decode("utf-8"),
        }

        with self.paths.trajectory.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

        return TrajectoryRecord(line_offset=line_offset, record_id=resolved_id)

    def _next_line_offset(self) -> int:
        if not self.paths.trajectory.exists():
            return 1

        with self.paths.trajectory.open(encoding="utf-8") as file:
            return sum(1 for _ in file) + 1


def _format_time(value: str | datetime | None) -> str:
    if value is None:
        return datetime.now(UTC).isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
