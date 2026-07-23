"""Public CLI for the Memorizer memory skill.

Memorizer agents call this file from shell. Durable writes are delegated to the
skill-local gate module, never to the foreground application CLI.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime

import gate


USAGE = """Usage:
  python scripts/memory.py inspect [--workspace PATH]
  python scripts/memory.py stage-episode --content TEXT --evidence trajectory.jsonl:Lx[-Ly] [--workspace PATH]
  python scripts/memory.py stage-semantic --content TEXT --episode E-0001 --status active [--trajectory-evidence trajectory.jsonl:Lx[-Ly]] [--workspace PATH]
  python scripts/memory.py revise-belief --memory-id S-0001 --status superseded_by:S-0002 [--workspace PATH]
  python scripts/memory.py stage-meta-harness --title TEXT --body TEXT [--workspace PATH]
  python scripts/memory.py commit [--workspace PATH]
  python scripts/memory.py clear-staging [--workspace PATH]
"""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0

    command = args.pop(0)
    workspace_arg = _workspace_from_args(args)
    _log_skill_event(command, "skill_command_start", workspace_arg)
    try:
        if command == "inspect":
            values = gate.parse_script_args(args, optional=["workspace"])
            print(json.dumps(gate.inspect(values["workspace"]), ensure_ascii=False, indent=2, sort_keys=True))
        elif command == "stage-episode":
            values = gate.parse_script_args(args, required=["content", "evidence"], optional=["workspace"])
            gate.stage_episode(str(values["content"]), str(values["evidence"]), values["workspace"])
        elif command == "stage-semantic":
            values = gate.parse_script_args(
                args,
                required=["content", "episode", "status"],
                optional=["trajectory_evidence", "workspace"],
            )
            gate.stage_semantic(
                str(values["content"]),
                str(values["episode"]),
                str(values["status"]),
                values["trajectory_evidence"],
                values["workspace"],
            )
        elif command == "revise-belief":
            values = gate.parse_script_args(args, required=["memory_id", "status"], optional=["workspace"])
            gate.revise_belief(str(values["memory_id"]), str(values["status"]), values["workspace"])
        elif command == "stage-meta-harness":
            values = gate.parse_script_args(args, required=["title", "body"], optional=["workspace"])
            gate.stage_meta_harness(str(values["title"]), str(values["body"]), values["workspace"])
        elif command == "commit":
            values = gate.parse_script_args(args, optional=["workspace"])
            result = gate.commit(values["workspace"])
            print(json.dumps({"version": result.version, "committed": result.committed}))
        elif command == "clear-staging":
            values = gate.parse_script_args(args, optional=["workspace"])
            gate.clear_staging(values["workspace"])
        else:
            raise SystemExit(f"unknown command: {command}\n\n{USAGE}")
    except BaseException as exc:
        _log_skill_event(command, "skill_command_error", workspace_arg, error=str(exc))
        raise
    _log_skill_event(command, "skill_command_success", workspace_arg)
    return 0


def _workspace_from_args(args: list[str]) -> str | None:
    for index, token in enumerate(args):
        if token == "--workspace" and index + 1 < len(args):
            return args[index + 1]
    return None


def _log_skill_event(command: str, event: str, workspace_arg: str | None, **fields: object) -> None:
    if os.getenv("DEBUG", "").strip().lower() not in {"1", "true", "ture", "yes", "on"}:
        return
    try:
        ws = gate.workspace(workspace_arg)
        path = ws.root / "runtime" / "logs" / "skill.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "agent": "skill",
            "event": event,
            "command": command,
            **fields,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        return


if __name__ == "__main__":
    sys.exit(main())
