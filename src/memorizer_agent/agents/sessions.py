from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import json
import threading
import time
import uuid

from memorizer_agent.agents.factory import (
    MEMORY_SKILL_SCRIPT,
    create_memorizer_agent,
    create_reasoner_agent,
)
from memorizer_agent.paths import WorkspacePaths
from memorizer_agent.runtime.context import MemoryContextInjector
from memorizer_agent.runtime.observability import (
    TokenUsage,
    create_trace_sink,
    extract_token_usage,
    merge_usage,
)
from memorizer_agent.runtime.trajectory import TrajectoryRecorder


class InvokableAgent(Protocol):
    def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        ...


class EchoReasonerAgent:
    def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        messages = payload.get("messages", [])
        user_messages = [message for message in messages if message.get("role") == "user"]
        content = user_messages[-1]["content"] if user_messages else ""
        return {"messages": [{"role": "assistant", "content": f"Reasoner placeholder: {content}"}]}


class SilentMemorizerAgent:
    def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"messages": [{"role": "assistant", "content": ""}]}


@dataclass(frozen=True)
class AgentRunResult:
    message: str
    completed: bool


def create_reasoner_session(paths: WorkspacePaths, model: Any) -> ReasonerSession:
    return ReasonerSession(paths=paths, agent=create_reasoner_agent(paths, model))


def create_background_memorizer_plugin(paths: WorkspacePaths, model: Any) -> BackgroundMemorizerPlugin:
    return BackgroundMemorizerPlugin(paths=paths, agent=create_memorizer_agent(paths, model))


@dataclass
class ReasonerSession:
    paths: WorkspacePaths
    agent: InvokableAgent
    messages: list[dict[str, str]] = field(default_factory=list)
    recorder: TrajectoryRecorder | None = None
    injector: MemoryContextInjector | None = None
    trace: Any | None = None

    def __post_init__(self) -> None:
        if self.recorder is None:
            self.recorder = TrajectoryRecorder(self.paths)
        if self.injector is None:
            self.injector = MemoryContextInjector(self.paths)
        if self.trace is None:
            self.trace = create_trace_sink(self.paths, "reasoner")

    def ask(self, user_message: str) -> str:
        turn_id = str(uuid.uuid4())
        assert self.recorder is not None
        assert self.injector is not None
        context_message = self.injector.build_context_message()
        assert self.trace is not None
        self.trace.event("memory_context", source=_memory_context_source(context_message))
        payload_messages = [
            {"role": "system", "content": context_message},
            *self.messages,
            {"role": "user", "content": user_message},
        ]
        self.recorder.record("User", user_message, record_id=f"{turn_id}:user")
        self.trace.event("agent_start")
        assistant_message = _run_agent_with_trace(
            self.agent,
            {"messages": payload_messages},
            trace=self.trace,
            agent_name="reasoner",
        ).message
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": assistant_message})
        self.recorder.record("Agent", assistant_message, record_id=f"{turn_id}:agent")
        self.injector.mark_consumed_latest()
        self.trace.event("agent_done")
        return assistant_message


@dataclass
class BackgroundMemorizerPlugin:
    paths: WorkspacePaths
    agent: InvokableAgent
    messages: list[dict[str, str]] = field(default_factory=list)
    last_observed_line: int = 0
    loop_control_index: int | None = None
    trace: Any | None = None
    min_interval_seconds: float = 2.0
    clock: Any = time.monotonic
    last_run_at: float | None = None

    def __post_init__(self) -> None:
        if self.trace is None:
            self.trace = create_trace_sink(self.paths, "memorizer", console=False)

    def observe_new_trajectory(self) -> bool:
        latest_line = _count_lines(self.paths.trajectory)
        if latest_line <= self.last_observed_line:
            return False

        start_line = self.last_observed_line + 1
        if _window_is_user_messages_only(self.paths.trajectory, start_line, latest_line):
            return False

        now = float(self.clock())
        if self.last_run_at is not None and now - self.last_run_at < self.min_interval_seconds:
            assert self.trace is not None
            self.trace.event(
                "loop_rate_limited",
                line_window=_line_window(start_line, latest_line),
                min_interval_seconds=self.min_interval_seconds,
            )
            return False

        line_window = _line_window(start_line, latest_line)
        assert self.trace is not None
        self.trace.event("loop_observed", line_window=line_window)
        prompt = self._build_loop_prompt(start_line, latest_line)
        self._merge_loop_control_message(prompt)
        _run_agent_with_trace(
            self.agent,
            {"messages": self.messages},
            trace=self.trace,
            agent_name="memorizer",
        )
        self.last_observed_line = latest_line
        self.last_run_at = now
        return True

    def _build_loop_prompt(self, start_line: int, latest_line: int) -> str:
        line_window = _line_window(start_line, latest_line)
        trajectory_offset = start_line - 1
        trajectory_limit = latest_line - start_line + 1
        return (
            "Background Memorizer loop tick. Do not respond to the user.\n"
            "Apply the system prompt silently. Process only the new trajectory window.\n\n"
            "File reads: use read_file with workspace-relative paths exactly as shown below.\n"
            "Shell commands already start in the workspace root; do not prepend Set-Location or pass Windows absolute paths.\n"
            "Do not use Get-Content/type/cat/more for these project files unless read_file failed.\n"
            "Required read_file calls before any shell command:\n"
            f"1. read_file data/trajectory/trajectory.jsonl offset={trajectory_offset} limit={trajectory_limit}\n"
            "2. read_file memory/Memories.md; if the tool reports not found, continue because commit can create it.\n"
            "3. In a fresh or uncertain context, read_file skills/memorizer-memory/SKILL.md before using the skill CLI.\n"
            "Trajectory: data/trajectory/trajectory.jsonl\n"
            "Memories: memory/Memories.md\n"
            f"New trajectory lines: {line_window}\n"
            "Do not infer the new content from this prompt; read the trajectory file.\n\n"
            "Flow:\n"
            f"1. Run exactly once: python {MEMORY_SKILL_SCRIPT} inspect --workspace .\n"
            "2. If durable memory is needed, use skill CLI commands: stage-episode, stage-semantic, revise-belief, commit.\n"
            "3. If nothing durable is needed, run no write commands.\n"
            "4. Final assistant message must be exactly one line starting with MEMORY_COMMITTED: or NO_DURABLE_MEMORY:."
        )

    def _merge_loop_control_message(self, prompt: str) -> None:
        if self.loop_control_index is None:
            self.messages.append({"role": "user", "content": prompt})
            self.loop_control_index = len(self.messages) - 1
            return

        self.messages[self.loop_control_index] = {"role": "user", "content": prompt}


def _last_assistant_content(response: dict[str, Any]) -> str:
    for message in reversed(response.get("messages", [])):
        if isinstance(message, dict):
            if message.get("role") == "assistant":
                return str(message.get("content", ""))
            continue
        if getattr(message, "type", None) == "ai":
            return str(getattr(message, "content", ""))
    return ""


class BackgroundMemorizerLoop:
    def __init__(self, plugin: BackgroundMemorizerPlugin, interval_seconds: float = 1.0) -> None:
        self.plugin = plugin
        self.interval_seconds = interval_seconds
        self.last_error: Exception | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="memorizer-loop", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.plugin.observe_new_trajectory()
            except Exception as exc:  # noqa: BLE001
                self.last_error = exc
                trace = getattr(self.plugin, "trace", None)
                if trace is not None:
                    trace.event("background_error", error=str(exc))
            self._stop_event.wait(self.interval_seconds)


def _count_lines(path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as file:
        return sum(1 for _ in file)


def _line_window(start_line: int, latest_line: int) -> str:
    return f"L{start_line}" if start_line == latest_line else f"L{start_line}-L{latest_line}"


def _window_is_user_messages_only(path, start_line: int, latest_line: int) -> bool:
    records = _trajectory_records(path, start_line, latest_line)
    return bool(records) and all(record.get("speaker") == "User" for record in records)


def _trajectory_records(path, start_line: int, latest_line: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line_number < start_line:
                continue
            if line_number > latest_line:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                records.append({"speaker": "unknown"})
                continue
            records.append(record if isinstance(record, dict) else {"speaker": "unknown"})
    return records


def _memory_context_source(context_message: str) -> str:
    if context_message.startswith("Latest memory diff"):
        return "memory/diffs/latest.diff.md"
    if context_message.startswith("Current memories"):
        return "memory/Memories.md"
    return "none"


def _run_agent_with_trace(
    agent: InvokableAgent,
    payload: dict[str, Any],
    *,
    trace: Any,
    agent_name: str,
) -> AgentRunResult:
    response = agent.invoke(payload)
    usage = _usage_from_response(response, agent_name)
    if usage is not None:
        trace.token_usage(usage)
    message = _last_assistant_content(response)
    return AgentRunResult(message=message, completed=True)


def _usage_from_response(response: dict[str, Any], agent_name: str) -> TokenUsage | None:
    usage: TokenUsage | None = None
    for message in response.get("messages", []):
        usage = merge_usage(usage, extract_token_usage(message, agent_name))
    return usage
