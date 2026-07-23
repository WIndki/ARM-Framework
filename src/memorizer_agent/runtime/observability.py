from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import json
import os
import threading

from memorizer_agent.paths import WorkspacePaths


@dataclass(frozen=True)
class TokenUsage:
    agent: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    cached_output_tokens: int = 0


class JsonlLogger:
    def __init__(self, paths: WorkspacePaths) -> None:
        self.paths = paths
        self._lock = threading.Lock()

    def log_agent_event(
        self,
        agent: str,
        event: str,
        *,
        log_path: Path | None = None,
        **fields: Any,
    ) -> None:
        path = log_path or self._agent_log_path(agent)
        self._append(path, {"agent": agent, "event": event, **fields})

    def log_token_usage(self, usage: TokenUsage) -> None:
        self._append(
            self.paths.token_usage_log,
            {
                "agent": usage.agent,
                "event": "token_usage",
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "cached_output_tokens": usage.cached_output_tokens,
            },
            include_timestamp=False,
        )

    def _agent_log_path(self, agent: str) -> Path:
        if agent == "reasoner":
            return self.paths.reasoner_log
        if agent == "memorizer":
            return self.paths.memorizer_log
        if agent == "skill":
            return self.paths.skill_log
        return self.paths.logs / f"{agent}.jsonl"

    def _append(self, path: Path, record: dict[str, Any], *, include_timestamp: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if include_timestamp:
            record = {"timestamp": datetime.now(UTC).isoformat(), **record}
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


class TraceSink:
    def __init__(
        self,
        paths: WorkspacePaths,
        agent: str,
        *,
        console: bool = True,
        log_path: Path | None = None,
    ) -> None:
        self.paths = paths
        self.agent = agent
        self.console = console
        self.log_path = log_path
        self.logger = JsonlLogger(paths)

    def event(self, event: str, **fields: Any) -> None:
        self.logger.log_agent_event(
            self.agent,
            event,
            log_path=self.log_path,
            **fields,
        )
        if self.console:
            line = self._format_console_event(event, fields)
            if line:
                print(line, flush=True)

    def token(self, content: str) -> None:
        if not content:
            return
        self.logger.log_agent_event(
            self.agent,
            "token",
            log_path=self.log_path,
            content=content,
        )
        if self.console:
            print(content, end="", flush=True)

    def token_usage(self, usage: TokenUsage) -> None:
        self.logger.log_token_usage(usage)
        self.event(
            "token_usage",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            cached_output_tokens=usage.cached_output_tokens,
        )

    def _format_console_event(self, event: str, fields: dict[str, Any]) -> str:
        prefix = f"[{self.agent}]"
        if event == "memory_context":
            source = fields.get("source", "memory context")
            return f"{prefix} inject memory context: {source}"
        if event == "file_read":
            return f"{prefix} read: {fields.get('path')}"
        if event == "file_write":
            return f"{prefix} write: {fields.get('path')}"
        if event == "file_edit":
            return f"{prefix} edit: {fields.get('path')}"
        if event == "file_write_blocked":
            return f"{prefix} blocked write: {fields.get('path')} ({fields.get('error')})"
        if event == "shell_command":
            return f"{prefix} shell: {fields.get('command')}"
        if event == "shell_result":
            return f"{prefix} shell result: exit={fields.get('exit_code')}"
        if event == "loop_observed":
            return f"{prefix} observed trajectory: {fields.get('line_window')}"
        if event == "agent_start":
            return f"{prefix} agent start"
        if event == "agent_done":
            return f"{prefix} agent done"
        if event == "tool_call":
            name = fields.get("name") or fields.get("tool") or "tool"
            return f"{prefix} tool: {name}"
        if event == "tool_result":
            name = fields.get("name") or fields.get("tool") or "tool"
            return f"{prefix} tool result: {name}"
        if event == "stream_event":
            name = fields.get("name") or fields.get("event") or "event"
            return f"{prefix} event: {name}"
        if event == "token_usage":
            return (
                f"{prefix} token usage: input={fields.get('input_tokens', 0)} "
                f"output={fields.get('output_tokens', 0)} total={fields.get('total_tokens', 0)} "
                f"cached_input={fields.get('cached_input_tokens', 0)}"
            )
        if event == "background_error":
            return f"{prefix} background error: {fields.get('error')}"
        return ""


class NullTraceSink:
    def event(self, event: str, **fields: Any) -> None:
        return

    def token(self, content: str) -> None:
        return

    def token_usage(self, usage: TokenUsage) -> None:
        return


def debug_enabled() -> bool:
    return os.getenv("DEBUG", "").strip().lower() in {
        "1",
        "true",
        "ture",
        "yes",
        "on",
    }


def create_trace_sink(
    paths: WorkspacePaths,
    agent: str,
    *,
    console: bool = True,
    log_path: Path | None = None,
) -> TraceSink | NullTraceSink:
    if not debug_enabled():
        return NullTraceSink()
    return TraceSink(paths, agent, console=console, log_path=log_path)


def extract_token_usage(message: Any, agent: str) -> TokenUsage | None:
    raw = getattr(message, "usage_metadata", None)
    if raw:
        return TokenUsage(
            agent=agent,
            input_tokens=int(raw.get("input_tokens", 0) or 0),
            output_tokens=int(raw.get("output_tokens", 0) or 0),
            total_tokens=int(raw.get("total_tokens", 0) or 0),
            cached_input_tokens=_cached_input_from_usage_metadata(raw),
            cached_output_tokens=_cached_output_from_usage_metadata(raw),
        )

    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or {}
    if token_usage:
        input_tokens = int(token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)) or 0)
        output_tokens = int(token_usage.get("completion_tokens", token_usage.get("output_tokens", 0)) or 0)
        total_tokens = int(token_usage.get("total_tokens", input_tokens + output_tokens) or 0)
        return TokenUsage(
            agent=agent,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=_cached_input_from_response_metadata(token_usage),
            cached_output_tokens=_cached_output_from_response_metadata(token_usage),
        )

    return None


def merge_usage(left: TokenUsage | None, right: TokenUsage | None) -> TokenUsage | None:
    if right is None:
        return left
    if left is None:
        return right
    return TokenUsage(
        agent=left.agent,
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        cached_input_tokens=left.cached_input_tokens + right.cached_input_tokens,
        cached_output_tokens=left.cached_output_tokens + right.cached_output_tokens,
    )


def _cached_input_from_usage_metadata(raw: Any) -> int:
    details = raw.get("input_token_details") or raw.get("prompt_token_details") or {}
    return int(
        details.get("cache_read")
        or details.get("cached_tokens")
        or details.get("cache_read_tokens")
        or raw.get("cached_input_tokens")
        or 0
    )


def _cached_output_from_usage_metadata(raw: Any) -> int:
    details = raw.get("output_token_details") or raw.get("completion_token_details") or {}
    return int(details.get("cache_read") or details.get("cached_tokens") or raw.get("cached_output_tokens") or 0)


def _cached_input_from_response_metadata(token_usage: dict[str, Any]) -> int:
    details = (
        token_usage.get("prompt_tokens_details")
        or token_usage.get("input_tokens_details")
        or token_usage.get("input_token_details")
        or {}
    )
    return int(
        details.get("cached_tokens")
        or details.get("cache_read")
        or token_usage.get("cached_input_tokens")
        or 0
    )


def _cached_output_from_response_metadata(token_usage: dict[str, Any]) -> int:
    details = (
        token_usage.get("completion_tokens_details")
        or token_usage.get("output_tokens_details")
        or token_usage.get("output_token_details")
        or {}
    )
    return int(
        details.get("cached_tokens")
        or details.get("cache_read")
        or token_usage.get("cached_output_tokens")
        or 0
    )
