import json
import re
from pathlib import Path


SNAPSHOT_PATTERN = re.compile(r"^memories_v(\d+)\.md$")


def read_consumption(path: Path) -> int:
    if not path.exists():
        return 0

    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data.get("memory_version", 0))


def write_consumption(path: Path, memory_version: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"memory_version": memory_version}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def latest_snapshot_version(snapshots_path: Path) -> int:
    if not snapshots_path.exists():
        return 0

    versions = []
    for snapshot in snapshots_path.iterdir():
        match = SNAPSHOT_PATTERN.match(snapshot.name)
        if match:
            versions.append(int(match.group(1)))

    return max(versions, default=0)
