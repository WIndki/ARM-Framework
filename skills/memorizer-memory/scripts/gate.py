"""Skill-local memory gate and durable writer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, NamedTuple


TRAJECTORY_RE = re.compile(r"^trajectory\.jsonl:L(?P<start>[1-9]\d*)(?:-L(?P<end>[1-9]\d*))?$")
MEMORY_RE = re.compile(r"^Memories\.md:(?P<memory_id>E-\d{4})@L(?P<line>[1-9]\d*)$")
PLAIN_STATUSES = {"active", "downgraded", "stale"}
SEMANTIC_DIFF_OPS = {"append_semantic", "revise_semantic_status"}


class TrajectoryRef(NamedTuple):
    start_line: int
    end_line: int


class MemoryRef(NamedTuple):
    memory_id: str
    line: int


class Episode(NamedTuple):
    memory_id: str
    content: str
    evidence: str


class Semantic(NamedTuple):
    memory_id: str
    status: str
    content: str
    evidence: str


class Projection(NamedTuple):
    version: int
    episodes: list[Episode]
    semantics: list[Semantic]


class Rendered(NamedTuple):
    text: str
    line_offsets: dict[str, int]


class CommitResult(NamedTuple):
    version: int
    committed: int


class Workspace(NamedTuple):
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
    def consumption(self) -> Path:
        return self.root / "runtime" / "reasoner_consumption.json"

    @property
    def pending(self) -> Path:
        return self.staging / "pending.jsonl"

    def ensure(self) -> None:
        for path in [
            self.trajectory.parent,
            self.memories.parent,
            self.ledger.parent,
            self.snapshots,
            self.diffs,
            self.staging,
            self.archive,
            self.consumption.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def workspace(workspace_arg: str | None = None) -> Workspace:
    value = workspace_arg or os.environ.get("MEMORIZER_WORKSPACE") or os.environ.get("workspace") or "."
    root = Path(value).resolve()
    root.mkdir(parents=True, exist_ok=True)
    ws = Workspace(root)
    _assert_paths_stay_inside_workspace(ws)
    return ws


def inspect(workspace_arg: str | None = None) -> dict[str, object]:
    ws = workspace(workspace_arg)
    pending = _read_pending(ws)
    projection = _project(_read_ledger(ws.ledger))
    return {"pending": pending, "pending_count": len(pending), "version": projection.version}


def stage_episode(content: str, evidence: str, workspace_arg: str | None = None) -> None:
    ws = workspace(workspace_arg)
    ws.ensure()
    _append_pending(ws, {"op": "append_episode", "content": content, "evidence": evidence})


def stage_semantic(
    content: str,
    episode_id: str,
    status: str,
    trajectory_evidence: str | None = None,
    workspace_arg: str | None = None,
) -> None:
    ws = workspace(workspace_arg)
    ws.ensure()
    _append_pending(
        ws,
        {
            "op": "append_semantic",
            "content": content,
            "episode_id": episode_id,
            "status": status,
            "trajectory_evidence": trajectory_evidence,
        },
    )


def revise_belief(memory_id: str, status: str, workspace_arg: str | None = None) -> None:
    ws = workspace(workspace_arg)
    ws.ensure()
    _append_pending(ws, {"op": "revise_semantic_status", "memory_id": memory_id, "status": status})


def stage_meta_harness(title: str, body: str, workspace_arg: str | None = None) -> None:
    ws = workspace(workspace_arg)
    ws.ensure()
    _append_pending(ws, {"op": "stage_meta_harness", "title": title, "body": body})


def clear_staging(workspace_arg: str | None = None) -> None:
    ws = workspace(workspace_arg)
    if ws.pending.exists():
        ws.pending.unlink()


def commit(workspace_arg: str | None = None) -> CommitResult:
    ws = workspace(workspace_arg)
    ws.ensure()
    pending = _read_pending(ws)
    ledger = _read_ledger(ws.ledger)
    projection = _project(ledger)
    next_version = projection.version + 1
    commit_records = _validate_and_build(ws, pending, projection, next_version)
    if not commit_records:
        return CommitResult(version=projection.version, committed=0)

    records_without_metadata = [*ledger, *commit_records]
    rendered = _render(_project(records_without_metadata))
    snapshot = ws.snapshots / f"memories_v{next_version:06d}.md"
    since_version = _read_consumption(ws.consumption)
    latest_diff = _render_diff(records_without_metadata, since_version)
    metadata = [
        {"version": next_version, "op": "create_snapshot", "path": snapshot.name},
        {
            "version": next_version,
            "op": "create_diff",
            "path": "latest.diff.md",
            "since_version": since_version,
        },
    ]
    durable_records = [*records_without_metadata, *metadata]
    ledger_text = "".join(_compact(record) + "\n" for record in durable_records)
    _publish(ws, ledger_text, rendered.text + "\n", snapshot, latest_diff)
    _archive_pending(ws, next_version)
    return CommitResult(version=next_version, committed=len(commit_records))


def parse_evidence(evidence: str) -> list[TrajectoryRef | MemoryRef]:
    refs: list[TrajectoryRef | MemoryRef] = []
    for raw in evidence.split(";"):
        part = raw.strip()
        if not part:
            raise ValueError("unsupported evidence syntax")
        trajectory = TRAJECTORY_RE.fullmatch(part)
        if trajectory:
            start = int(trajectory.group("start"))
            end = int(trajectory.group("end") or start)
            if end < start:
                raise ValueError("trajectory evidence range end before start")
            refs.append(TrajectoryRef(start, end))
            continue
        memory = MEMORY_RE.fullmatch(part)
        if memory:
            refs.append(MemoryRef(memory.group("memory_id"), int(memory.group("line"))))
            continue
        raise ValueError(f"unsupported evidence syntax: {part}")
    return refs


def validate_trajectory_refs(refs: list[TrajectoryRef], trajectory: Path) -> None:
    lines = trajectory.read_text(encoding="utf-8").splitlines()
    for ref in refs:
        for line in range(ref.start_line, ref.end_line + 1):
            if line > len(lines):
                raise ValueError(f"missing trajectory line {line}")
            if not lines[line - 1].strip():
                raise ValueError(f"blank trajectory line {line}")


def render_memories(projection: Projection) -> Rendered:
    return _render(projection)


def parse_script_args(
    argv: list[str],
    *,
    required: list[str] | None = None,
    optional: list[str] | None = None,
) -> dict[str, str | None]:
    required = required or []
    optional = optional or []
    allowed = {*required, *optional}
    values: dict[str, str | None] = {name: None for name in allowed}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            raise SystemExit(f"unexpected argument: {token}")
        name = token[2:].replace("-", "_")
        if name not in allowed:
            raise SystemExit(f"unknown option: {token}")
        if index + 1 >= len(argv):
            raise SystemExit(f"missing value for {token}")
        values[name] = argv[index + 1]
        index += 2
    missing = [name for name in required if values.get(name) is None]
    if missing:
        raise SystemExit(f"missing required option(s): {', '.join('--' + item.replace('_', '-') for item in missing)}")
    return values


def _validate_and_build(
    ws: Workspace,
    pending: list[dict[str, Any]],
    projection: Projection,
    version: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    committed_episodes = {episode.memory_id for episode in projection.episodes}
    existing_episodes = set(committed_episodes)
    existing_semantics = {semantic.memory_id for semantic in projection.semantics}
    next_episode = len(existing_episodes) + 1
    next_semantic = len(existing_semantics) + 1
    working_records: list[dict[str, object]] = []

    for stage in pending:
        op = stage["op"]
        if op == "append_episode":
            _validate_episode_evidence(ws, str(stage["evidence"]))
            memory_id = f"E-{next_episode:04d}"
            next_episode += 1
            existing_episodes.add(memory_id)
            record = {
                "version": version,
                "op": "append_episode",
                "memory_id": memory_id,
                "content": str(stage["content"]),
                "evidence": str(stage["evidence"]),
            }
        elif op == "append_semantic":
            status = str(stage["status"])
            episode_id = str(stage["episode_id"])
            if episode_id not in committed_episodes:
                raise ValueError(f"missing committed episode {episode_id}")
            trajectory_evidence = stage.get("trajectory_evidence")
            if trajectory_evidence is not None:
                _validate_episode_evidence(ws, str(trajectory_evidence))
            memory_id = f"S-{next_semantic:04d}"
            _validate_status(status, existing_semantics | {memory_id})
            next_semantic += 1
            existing_semantics.add(memory_id)
            record = {
                "version": version,
                "op": "append_semantic",
                "memory_id": memory_id,
                "status": status,
                "content": str(stage["content"]),
                "evidence": _semantic_evidence(projection, working_records, episode_id, trajectory_evidence),
            }
        elif op == "revise_semantic_status":
            status = str(stage["status"])
            memory_id = str(stage["memory_id"])
            if memory_id not in existing_semantics:
                raise ValueError(f"missing semantic {memory_id}")
            _validate_status(status, existing_semantics)
            record = {
                "version": version,
                "op": "revise_semantic_status",
                "memory_id": memory_id,
                "status": status,
            }
        elif op == "stage_meta_harness":
            record = {
                "version": version,
                "op": "stage_meta_harness",
                "title": str(stage["title"]),
                "body": str(stage["body"]),
            }
        else:
            raise ValueError(f"unsupported stage op: {op}")
        records.append(record)
        working_records.append(record)
    return records


def _semantic_evidence(
    projection: Projection,
    working_records: list[dict[str, object]],
    episode_id: str,
    trajectory_evidence: object,
) -> str:
    working_projection = _project(
        [
            *[
                {
                    "version": projection.version,
                    "op": "append_episode",
                    "memory_id": episode.memory_id,
                    "content": episode.content,
                    "evidence": episode.evidence,
                }
                for episode in projection.episodes
            ],
            *[
                {
                    "version": projection.version,
                    "op": "append_semantic",
                    "memory_id": semantic.memory_id,
                    "status": semantic.status,
                    "content": semantic.content,
                    "evidence": semantic.evidence,
                }
                for semantic in projection.semantics
            ],
            *working_records,
        ]
    )
    rendered = _render(working_projection)
    evidence = f"Memories.md:{episode_id}@L{rendered.line_offsets[episode_id]}"
    if trajectory_evidence is not None:
        evidence = f"{evidence}; {trajectory_evidence}"
    return evidence


def _validate_episode_evidence(ws: Workspace, evidence: str) -> None:
    refs = parse_evidence(evidence)
    trajectory_refs = [ref for ref in refs if isinstance(ref, TrajectoryRef)]
    if not trajectory_refs:
        raise ValueError("episode evidence must include trajectory ref")
    validate_trajectory_refs(trajectory_refs, ws.trajectory)


def _validate_status(status: str, known_semantics: set[str]) -> None:
    if status in PLAIN_STATUSES:
        return
    if status.startswith("superseded_by:S-") or status.startswith("contradicted_by:S-"):
        target = status.rsplit(":", maxsplit=1)[1]
        if len(target) == 6 and target[2:].isdigit():
            if target not in known_semantics:
                raise ValueError(f"missing semantic status target {target}")
            return
    raise ValueError(f"unsupported semantic status: {status}")


def _render(projection: Projection) -> Rendered:
    lines = ["# Memories", "", "## Episodic Memory"]
    offsets: dict[str, int] = {}
    for episode in projection.episodes:
        _require_one_line(episode.memory_id, "episode memory_id")
        _require_one_line(episode.content, "episode content")
        _require_one_line(episode.evidence, "episode evidence")
        offsets[episode.memory_id] = len(lines) + 1
        lines.append(f"[{episode.memory_id}] {episode.content} [Evidence: {episode.evidence}]")
    lines.extend(["", "## Semantic Memory"])
    for semantic in projection.semantics:
        _require_one_line(semantic.memory_id, "semantic memory_id")
        _require_one_line(semantic.status, "semantic status")
        _require_one_line(semantic.content, "semantic content")
        _require_one_line(semantic.evidence, "semantic evidence")
        offsets[semantic.memory_id] = len(lines) + 1
        lines.append(
            f"[{semantic.memory_id} {semantic.status}] {semantic.content} "
            f"[Evidence: {semantic.evidence}]"
        )
    return Rendered("\n".join(lines), offsets)


def _render_diff(records: list[dict[str, object]], since_version: int) -> str:
    lines = [f"# Memory Diff Since v{since_version}", ""]
    semantics: dict[str, Semantic] = {}
    for record in records:
        op = record.get("op")
        if op not in SEMANTIC_DIFF_OPS:
            continue
        if op == "append_semantic":
            memory_id = str(record["memory_id"])
            semantics[memory_id] = Semantic(
                memory_id,
                str(record["status"]),
                str(record["content"]),
                str(record["evidence"]),
            )
        elif op == "revise_semantic_status":
            memory_id = str(record["memory_id"])
            semantic = semantics.get(memory_id)
            if semantic is not None:
                semantics[memory_id] = Semantic(
                    semantic.memory_id,
                    str(record["status"]),
                    semantic.content,
                    semantic.evidence,
                )
        if int(record.get("version", 0)) <= since_version:
            continue
        if op == "append_semantic":
            lines.append(
                f"+ [SEMANTIC {record['status']}] "
                f"[{record['memory_id']} {record['status']}] {record['content']} "
                f"[Evidence: {record['evidence']}]"
            )
        elif op == "revise_semantic_status":
            semantic = semantics[str(record["memory_id"])]
            lines.append(
                f"~ [SEMANTIC {semantic.status}] "
                f"[{semantic.memory_id} {semantic.status}] {semantic.content} "
                f"[Evidence: {semantic.evidence}]"
            )
    return "\n".join(lines) + "\n"


def _project(records: list[dict[str, Any]]) -> Projection:
    episodes: list[Episode] = []
    semantics: dict[str, Semantic] = {}
    semantic_order: list[str] = []
    version = 0
    for record in records:
        version = max(version, int(record.get("version", 0)))
        op = record.get("op")
        if op == "append_episode":
            episodes.append(Episode(str(record["memory_id"]), str(record["content"]), str(record["evidence"])))
        elif op == "append_semantic":
            memory_id = str(record["memory_id"])
            semantics[memory_id] = Semantic(
                memory_id,
                str(record["status"]),
                str(record["content"]),
                str(record["evidence"]),
            )
            semantic_order.append(memory_id)
        elif op == "revise_semantic_status":
            memory_id = str(record["memory_id"])
            semantic = semantics[memory_id]
            semantics[memory_id] = Semantic(
                semantic.memory_id,
                str(record["status"]),
                semantic.content,
                semantic.evidence,
            )
    return Projection(version, episodes, [semantics[memory_id] for memory_id in semantic_order])


def _read_pending(ws: Workspace) -> list[dict[str, Any]]:
    if not ws.pending.exists():
        return []
    return [json.loads(line) for line in ws.pending.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_consumption(path: Path) -> int:
    if not path.exists():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8")).get("memory_version", 0))


def _append_pending(ws: Workspace, record: dict[str, object]) -> None:
    ws.staging.mkdir(parents=True, exist_ok=True)
    with ws.pending.open("a", encoding="utf-8") as handle:
        handle.write(_compact(record) + "\n")


def _publish(ws: Workspace, ledger_text: str, memory_text: str, snapshot: Path, diff_text: str) -> None:
    ledger_tmp = ws.ledger.with_suffix(".jsonl.tmp")
    memories_tmp = ws.memories.with_suffix(".md.tmp")
    snapshot_tmp = snapshot.with_suffix(".md.tmp")
    diff_path = ws.diffs / "latest.diff.md"
    diff_tmp = diff_path.with_suffix(".diff.md.tmp")
    ledger_tmp.write_text(ledger_text, encoding="utf-8")
    memories_tmp.write_text(memory_text, encoding="utf-8")
    snapshot_tmp.write_text(memory_text, encoding="utf-8")
    diff_tmp.write_text(diff_text, encoding="utf-8")
    targets = [
        (memories_tmp, ws.memories),
        (snapshot_tmp, snapshot),
        (diff_tmp, diff_path),
        (ledger_tmp, ws.ledger),
    ]
    backups: list[tuple[Path, Path | None]] = []
    replaced: list[Path] = []
    try:
        for source, target in targets:
            backup = None
            if target.exists():
                backup = target.with_name(f"{target.name}.bak")
                if backup.exists():
                    backup.unlink()
                shutil.copy2(target, backup)
            backups.append((target, backup))
            os.replace(source, target)
            replaced.append(target)
    except Exception:
        replaced_targets = set(replaced)
        for target in reversed(replaced):
            if target.exists():
                target.unlink()
        for target, backup in reversed(backups):
            if backup is not None and backup.exists() and target in replaced_targets:
                os.replace(backup, target)
            elif backup is not None and backup.exists():
                backup.unlink()
        for source, _ in targets:
            if source.exists():
                source.unlink()
        raise
    else:
        for _, backup in backups:
            if backup is not None and backup.exists():
                backup.unlink()


def _archive_pending(ws: Workspace, version: int) -> None:
    ws.archive.mkdir(parents=True, exist_ok=True)
    if ws.pending.exists():
        shutil.move(str(ws.pending), str(ws.archive / f"pending_v{version:06d}.jsonl"))


def _assert_paths_stay_inside_workspace(ws: Workspace) -> None:
    root = ws.root.resolve()
    for path in [
        ws.trajectory.parent,
        ws.memories.parent,
        ws.ledger.parent,
        ws.snapshots,
        ws.diffs,
        ws.staging,
        ws.archive,
        ws.consumption.parent,
    ]:
        if not path.resolve().is_relative_to(root):
            raise RuntimeError(f"path escapes workspace: {path}")


def _require_one_line(value: str, field: str) -> None:
    if "\n" in value or "\r" in value:
        raise ValueError(f"{field} must be one line")


def _compact(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))
