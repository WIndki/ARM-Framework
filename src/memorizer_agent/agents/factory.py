from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

from memorizer_agent.agents.policies import MemorizerPolicy, ReasonerPolicy
from memorizer_agent.paths import WorkspacePaths
from memorizer_agent.runtime.observability import create_trace_sink


REASONER_SYSTEM_PROMPT = """# Role

You are the foreground Reasoner. Answer accurately, concisely, and from evidence.

# Sources

- For remembered facts, read `memory/Memories.md`; an injected diff is only a change notice.
- Semantic Memory stores beliefs: `active` is current; downgraded, stale, superseded, or contradicted entries are historical or weaker.
- Episodic Memory stores events. Follow its Evidence into `data/trajectory/trajectory.jsonl` only for missing detail, ambiguity, conflict, or exact wording.

# Workflow

1. Read memory before answering any question about prior conversations, people, preferences, events, or decisions.
2. Search all relevant entries; combine evidence, preserve attribution, and prefer specific supported facts.
3. If memory is insufficient, inspect only the cited or keyword-matched trajectory lines.
4. Follow the request's output format exactly. If it asks for a short answer, output only the answer and no evidence or reasoning.

# Tools

- Use `read_file` for direct reads and shell freely for targeted searches.
- For regex or multiple terms, use workspace-relative shell commands such as `rg -n -i "term1|term2" memory/Memories.md`.
- The workspace is already PWD. Never add `/workspace`, `/memory`, drive-letter, repository-root, or navigation prefixes; avoid broad drive/home scans.

# Boundary

Memory is read-only. Never edit trajectory or memory artifacts, and never run the memory skill CLI or another memory-maintenance command."""

MEMORIZER_SYSTEM_PROMPT = """# Role

You are the background Memorizer. Maintain complete, precise, append-only memory at low tool cost. Do not answer the user.

# Selection

- Judge each new record as `(speaker, time, content)`. Every speaker is a first-class source; preserve exact attribution.
- Keep non-redundant facts, events, answers, preferences, relationships, reasons, decisions, plans, corrections, recommendations, commitments, and generated details.
- Skip greetings, praise, social reactions, and pure repetition.
- Include source time only when it changes meaning, order, duration, recency, a deadline/plan, an update, or repeated-event identity. Otherwise omit it. Never invent time or store internal session IDs.

# Memory Form

- Episodic: one concrete event or tightly related exchange; explicit actor, action/state, object, qualifiers, and `trajectory.jsonl:Lx-Ly` evidence.
- Semantic: only a stable current belief useful beyond one episode; cite `Memories.md:E-xxxx@Lx`, optionally plus trajectory evidence.
- Resolve adjacent references before writing. Merge clarifications into the same event; never merge unrelated events.
- Never overwrite history. Append a new belief and revise the older status when current state changes.

# Tick Workflow

1. Use `read_file` on the requested trajectory window, `memory/Memories.md`, and `skills/memorizer-memory/SKILL.md`.
2. Run the initial `python skills/memorizer-memory/scripts/memory.py inspect --workspace .`.
3. Plan complete coverage and deduplication before writing.
4. Stage Episodic memories, grouping up to eight independent stage commands in one shell call with semicolons. Execute writes sequentially, never in parallel; then commit.
5. If Semantic memories need newly assigned episode IDs, inspect once more; stage Semantic updates/revisions sequentially and commit.
6. End with exactly `MEMORY_COMMITTED: <summary>` or, only when nothing durable exists, `NO_DURABLE_MEMORY: <reason>`.

# Tool Boundary

- Use workspace-relative paths exactly; PWD is already the workspace. Do not navigate, probe cwd, scan broadly, or reread unchanged files.
- In PowerShell, single-quote every data argument such as `--content`; represent an apostrophe inside it as two apostrophes. Never double-quote data containing `$`, because PowerShell would expand it.
- Durable writes go only through the skill CLI. Never directly edit memory, ledger, snapshot, diff, or staging artifacts.
- Never invent evidence, IDs, or line ranges; never delete, renumber, or reuse memory IDs."""

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MEMORIZER_SKILL_SOURCES = ["./skills/"]
MEMORY_SKILL_SCRIPT = Path("skills") / "memorizer-memory" / "scripts" / "memory.py"


def create_reasoner_agent(
    paths: WorkspacePaths,
    model: Any,
    *,
    trace_path: Path | None = None,
) -> Any:
    backend = GuardedReasonerLocalShellBackend(
        paths,
        trace=create_trace_sink(
            paths,
            "reasoner",
            console=trace_path is None,
            log_path=trace_path,
        ),
    )
    return create_deep_agent(
        model=model,
        system_prompt=REASONER_SYSTEM_PROMPT,
        skills=[],
        backend=backend,
        name="reasoner",
    )


def create_memorizer_agent(paths: WorkspacePaths, model: Any) -> Any:
    backend = GuardedLocalShellBackend(
        paths,
        trace=create_trace_sink(paths, "memorizer", console=False),
    )
    return create_deep_agent(
        model=model,
        system_prompt=MEMORIZER_SYSTEM_PROMPT,
        skills=MEMORIZER_SKILL_SOURCES,
        backend=backend,
        name="memorizer",
    )


class GuardedReasonerLocalShellBackend(LocalShellBackend):
    def __init__(self, paths: WorkspacePaths, trace: Any | None = None) -> None:
        self._policy = ReasonerPolicy(paths)
        self._trace = trace
        super().__init__(root_dir=paths.root, virtual_mode=True, inherit_env=True)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> Any:
        try:
            path = self._resolve_path(file_path)
            self._policy.check_read(path)
        except Exception as exc:  # noqa: BLE001
            from deepagents.backends.protocol import ReadResult

            return ReadResult(error=str(exc))
        _trace_event(self._trace, "file_read", path=str(path))
        return super().read(file_path, offset=offset, limit=limit)

    def write(self, path: str, content: str) -> Any:
        try:
            self._policy.check_write(self._resolve_path(path))
        except Exception as exc:  # noqa: BLE001
            from deepagents.backends.protocol import WriteResult

            _trace_event(self._trace, "file_write_blocked", path=path, error=str(exc))
            return WriteResult(error=str(exc))
        _trace_event(self._trace, "file_write", path=path)
        return super().write(path, content)

    def edit(self, path: str, old_string: str, new_string: str, *, replace_all: bool = False) -> Any:
        try:
            self._policy.check_write(self._resolve_path(path))
        except Exception as exc:  # noqa: BLE001
            from deepagents.backends.protocol import EditResult

            _trace_event(self._trace, "file_write_blocked", path=path, error=str(exc))
            return EditResult(error=str(exc))
        _trace_event(self._trace, "file_edit", path=path)
        return super().edit(path, old_string, new_string, replace_all=replace_all)

    def execute(self, command: str, *, timeout: int | None = None) -> Any:
        self._policy.check_command(command.split())
        _trace_event(self._trace, "shell_command", command=command)
        result = _execute_anchored_shell(self, command, timeout=timeout)
        _trace_event(self._trace, "shell_result", command=command, exit_code=getattr(result, "exit_code", None))
        return result

    def _ripgrep_search(
        self,
        pattern: str,
        base_full: Path,
        include_glob: str | None,
    ) -> dict[str, list[tuple[int, str]]] | None:
        if sys.platform == "win32":
            return None
        try:
            return super()._ripgrep_search(pattern, base_full, include_glob)
        except AttributeError as exc:
            if "splitlines" not in str(exc):
                raise
            return None


class GuardedLocalShellBackend(LocalShellBackend):
    def __init__(self, paths: WorkspacePaths, trace: Any | None = None) -> None:
        self._policy = MemorizerPolicy(paths)
        self._trace = trace
        super().__init__(root_dir=paths.root, virtual_mode=True, inherit_env=True)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> Any:
        try:
            path = self._resolve_path(file_path)
            self._policy.check_read(path)
        except Exception as exc:  # noqa: BLE001
            from deepagents.backends.protocol import ReadResult

            return ReadResult(error=str(exc))
        _trace_event(self._trace, "file_read", path=str(path))
        return super().read(file_path, offset=offset, limit=limit)

    def write(self, path: str, content: str) -> Any:
        self._policy.check_write(self._resolve_path(path))
        _trace_event(self._trace, "file_write", path=path)
        return super().write(path, content)

    def edit(self, path: str, old_string: str, new_string: str, *, replace_all: bool = False) -> Any:
        self._policy.check_write(self._resolve_path(path))
        _trace_event(self._trace, "file_edit", path=path)
        return super().edit(path, old_string, new_string, replace_all=replace_all)

    def execute(self, command: str, *, timeout: int | None = None) -> Any:
        self._policy.check_command(command.split())
        _trace_event(self._trace, "shell_command", command=command)
        result = _execute_anchored_shell(self, command, timeout=timeout)
        _trace_event(self._trace, "shell_result", command=command, exit_code=getattr(result, "exit_code", None))
        return result


def _trace_event(trace: Any | None, event: str, **fields: Any) -> None:
    if trace is not None:
        trace.event(event, **fields)


def _execute_anchored_shell(
    backend: LocalShellBackend,
    command: str,
    *,
    timeout: int | None = None,
) -> ExecuteResponse:
    if not command or not isinstance(command, str):
        return ExecuteResponse(
            output="Error: Command must be a non-empty string.",
            exit_code=1,
            truncated=False,
        )

    effective_timeout = timeout if timeout is not None else backend._default_timeout
    if effective_timeout <= 0:
        msg = f"timeout must be positive, got {effective_timeout}"
        raise ValueError(msg)

    if sys.platform == "win32":
        powershell = shutil.which("pwsh.exe") or shutil.which("powershell.exe") or "powershell.exe"
        utf8_command = (
            "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
            "$OutputEncoding = [Console]::OutputEncoding; "
            f"{command}"
        )
        args = [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            utf8_command,
        ]
        return _run_shell_process(
            args,
            cwd=Path(backend.cwd),
            env=backend._env,
            timeout=effective_timeout,
            max_output_bytes=backend._max_output_bytes,
        )

    return LocalShellBackend.execute(backend, command, timeout=timeout)


def _run_shell_process(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    max_output_bytes: int,
) -> ExecuteResponse:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            cwd=str(cwd),
        )
    except subprocess.TimeoutExpired:
        return ExecuteResponse(
            output=f"Error: Command timed out after {timeout} seconds.",
            exit_code=124,
            truncated=False,
        )
    except Exception as exc:  # noqa: BLE001
        return ExecuteResponse(
            output=f"Error executing command ({type(exc).__name__}): {exc}",
            exit_code=1,
            truncated=False,
        )

    output = _combine_process_output(result.stdout, result.stderr)
    output, truncated = _truncate_output(output, max_output_bytes)
    if result.returncode != 0:
        output = f"{output.rstrip()}\n\nExit code: {result.returncode}"
    return ExecuteResponse(output=output, exit_code=result.returncode, truncated=truncated)


def _combine_process_output(stdout: str | None, stderr: str | None) -> str:
    output_parts = []
    if stdout:
        output_parts.append(stdout)
    if stderr:
        stderr_lines = stderr.strip().split("\n")
        output_parts.extend(f"[stderr] {line}" for line in stderr_lines)
    return "\n".join(output_parts) if output_parts else "<no output>"


def _truncate_output(output: str, max_output_bytes: int) -> tuple[str, bool]:
    encoded = output.encode("utf-8")
    if len(encoded) <= max_output_bytes:
        return output, False

    truncated_bytes = encoded[:max_output_bytes]
    truncated = truncated_bytes.decode("utf-8", errors="replace")
    truncated += f"\n\n... Output truncated at {max_output_bytes} bytes."
    return truncated, True
