from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable
import json
import math
import os
import re
import string
import uuid

from memorizer_agent.agents.factory import create_memorizer_agent, create_reasoner_agent
from memorizer_agent.config import build_chat_model, load_config
from memorizer_agent.paths import WorkspacePaths
from memorizer_agent.runtime.observability import JsonlLogger, debug_enabled
from memorizer_agent.runtime.trajectory import TrajectoryRecorder


@dataclass(frozen=True)
class Message:
    speaker: str
    role: str
    content: str


@dataclass(frozen=True)
class Session:
    session_id: str
    date: str | None
    messages: list[Message]


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    answer: str
    group: str
    question_date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    sessions: list[Session]
    questions: list[Question]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeResult:
    label: str
    score: float
    reasoning: str
    raw: str


class EvaluationHarness:
    """Evaluate one sample with online, chronological session-batch ingestion."""

    def __init__(
        self,
        *,
        run_root: Path,
        benchmark: str,
        model: Any,
        memory_focus: str,
        question_prompt: Callable[[Question], str],
        score_question: Callable[[Question, str], dict[str, Any]],
        resume: bool,
        session_batch_size: int = 1,
        qa_workers: int = 1,
    ) -> None:
        self.run_root = run_root.resolve()
        self.benchmark = benchmark
        self.model = model
        self.memory_focus = memory_focus
        self.question_prompt = question_prompt
        self.score_question = score_question
        self.resume = resume
        if session_batch_size < 1:
            raise ValueError("session_batch_size must be at least 1")
        if qa_workers < 1:
            raise ValueError("qa_workers must be at least 1")
        self.session_batch_size = session_batch_size
        self.qa_workers = qa_workers
        self.debug = debug_enabled()

    def evaluate_sample(self, sample: Sample) -> dict[str, Any]:
        sample_root = self.run_root / "workspaces" / safe_id(sample.sample_id)
        sample_result_dir = self.run_root / "samples" / safe_id(sample.sample_id)
        result_path = sample_result_dir / "sample.json"
        if self.resume and result_path.exists():
            return read_json(result_path)
        if result_path.exists():
            raise FileExistsError(
                f"result already exists: {result_path}; use --resume or a new --run-name"
            )

        paths = WorkspacePaths(sample_root)
        state = self._load_or_create_state(paths, sample)
        self._ingest_sessions(paths, sample, state)

        question_jobs = [
            (
                paths,
                question,
                sample_result_dir
                / f"qa-{index:04d}-{safe_id(question.question_id)}.json",
                sample_result_dir
                / f"qa-{index:04d}-{safe_id(question.question_id)}.log.jsonl"
                if self.debug
                else None,
            )
            for index, question in enumerate(sample.questions, start=1)
        ]
        if self.qa_workers == 1:
            qa_results = [self._evaluate_question(*job) for job in question_jobs]
        else:
            with ThreadPoolExecutor(max_workers=self.qa_workers) as executor:
                qa_results = list(executor.map(lambda job: self._evaluate_question(*job), question_jobs))
        qa_results = [read_json(job[2]) for job in question_jobs]

        result = {
            "benchmark": self.benchmark,
            "sample_id": sample.sample_id,
            "workspace": str(sample_root),
            "session_count": len(sample.sessions),
            "session_batch_size": self.session_batch_size,
            "qa_workers": self.qa_workers,
            "trajectory_lines": int(state["trajectory_lines"]),
            "memory_lines": count_lines(paths.memories),
            "questions": qa_results,
            "metadata": sample.metadata,
        }
        write_json(result_path, result)
        return result

    def _load_or_create_state(
        self,
        paths: WorkspacePaths,
        sample: Sample,
    ) -> dict[str, Any]:
        state_path = paths.root / ".eval_state.json"
        fingerprint = sample_fingerprint(sample)
        if state_path.exists():
            if not self.resume:
                raise FileExistsError(
                    f"sample workspace already exists: {paths.root}; use --resume or a new run"
                )
            state = read_json(state_path)
            if state.get("sample_fingerprint") != fingerprint:
                raise ValueError(f"dataset sample changed since the run began: {sample.sample_id}")
            configured_batch_size = int(
                state.get("session_batch_size", self.session_batch_size)
            )
            if configured_batch_size != self.session_batch_size:
                raise ValueError(
                    "session batch size changed since the run began: "
                    f"{configured_batch_size} -> {self.session_batch_size}"
                )
            paths.ensure()
            return state

        paths.ensure()
        state = {
            "benchmark": self.benchmark,
            "sample_id": sample.sample_id,
            "sample_fingerprint": fingerprint,
            "session_batch_size": self.session_batch_size,
            "phase": "ingesting",
            "appended_session_count": 0,
            "memorized_session_count": 0,
            "trajectory_lines": 0,
            "pending_window": None,
        }
        write_json(state_path, state)
        return state

    def _ingest_sessions(
        self,
        paths: WorkspacePaths,
        sample: Sample,
        state: dict[str, Any],
    ) -> None:
        state_path = paths.root / ".eval_state.json"
        recorder = TrajectoryRecorder(paths)
        memorizer = create_memorizer_agent(paths, self.model)

        while int(state["memorized_session_count"]) < len(sample.sessions):
            paths.ensure()
            memorized = int(state["memorized_session_count"])
            appended = int(state["appended_session_count"])

            if appended == memorized:
                batch_end = min(appended + self.session_batch_size, len(sample.sessions))
                start_line = int(state["trajectory_lines"]) + 1
                for session in sample.sessions[appended:batch_end]:
                    for message in session.messages:
                        next_line = int(state["trajectory_lines"]) + 1
                        recorder.record(
                            message.speaker.strip() or normalize_speaker(message.role),
                            message.content,
                            time=session.date,
                            record_id=f"r{next_line:08d}",
                        )
                        state["trajectory_lines"] = next_line

                state["appended_session_count"] = batch_end
                state["pending_window"] = {
                    "session_start": appended,
                    "session_end": batch_end,
                    "start_line": start_line,
                    "end_line": int(state["trajectory_lines"]),
                }
                write_json(state_path, state)

            pending = state.get("pending_window")
            if not isinstance(pending, dict):
                raise RuntimeError("ingestion state has no pending trajectory window")
            if int(pending["end_line"]) < int(pending["start_line"]):
                state["memorized_session_count"] = int(state["appended_session_count"])
                state["pending_window"] = None
                write_json(state_path, state)
                continue

            self._assert_sample_workspace(paths)
            response = memorizer.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": build_memorizer_prompt(
                                window=pending,
                                memory_focus=self.memory_focus,
                            ),
                        }
                    ]
                },
                config={
                    "configurable": {
                        "thread_id": f"eval:{state['sample_fingerprint'][:16]}"
                    }
                },
            )
            final_message = last_assistant_content(response)
            if memory_completion_marker(final_message) is None:
                raise RuntimeError(
                    f"Memorizer did not complete {sample.sample_id} lines "
                    f"L{pending['start_line']}-L{pending['end_line']}: "
                    f"{final_message[:500]}"
                )

            state["memorized_session_count"] = int(state["appended_session_count"])
            state["pending_window"] = None
            state["phase"] = (
                "memory_complete"
                if int(state["memorized_session_count"]) == len(sample.sessions)
                else "ingesting"
            )
            write_json(state_path, state)

    @staticmethod
    def _assert_sample_workspace(paths: WorkspacePaths) -> None:
        required = (
            paths.trajectory,
            paths.memory_skill / "SKILL.md",
            paths.memory_skill / "scripts" / "memory.py",
        )
        missing = [path.relative_to(paths.root).as_posix() for path in required if not path.exists()]
        if missing:
            raise RuntimeError(
                "sample workspace is incomplete before memory ingestion: "
                + ", ".join(missing)
            )

    def _answer(
        self,
        paths: WorkspacePaths,
        question: Question,
        trace_path: Path | None,
    ) -> str:
        reasoner = create_reasoner_agent(paths, self.model, trace_path=trace_path)
        prompt = (
            f"{self.question_prompt(question)}\n\n"
            "Files in the current workspace:\n"
            "- Complete memory: memory/Memories.md\n"
            "- Source trajectory: data/trajectory/trajectory.jsonl\n"
            "Use these exact relative paths for read_file or shell commands such as rg."
            " Never use /workspace, /memory, drive-letter, or repository-root prefixes."
        )
        response = reasoner.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        answer = last_assistant_content(response).strip()
        if not answer:
            raise RuntimeError(f"Reasoner returned an empty answer for {question.question_id}")
        return answer

    def _evaluate_question(
        self,
        paths: WorkspacePaths,
        question: Question,
        cache_path: Path,
        trace_path: Path | None,
    ) -> dict[str, Any]:
        if self.resume and cache_path.exists():
            if trace_path is not None and not trace_path.exists():
                JsonlLogger(paths).log_agent_event(
                    "reasoner",
                    "qa_resume_hit",
                    log_path=trace_path,
                    question_id=question.question_id,
                )
            return read_json(cache_path)
        logger = JsonlLogger(paths) if trace_path is not None else None
        if logger is not None:
            logger.log_agent_event(
                "reasoner",
                "qa_start",
                log_path=trace_path,
                question_id=question.question_id,
            )
        try:
            prediction = self._answer(paths, question, trace_path)
            scores = self.score_question(question, prediction)
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.log_agent_event(
                    "reasoner",
                    "qa_error",
                    log_path=trace_path,
                    question_id=question.question_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise RuntimeError(
                f"question evaluation failed for {question.question_id}: {exc}"
            ) from exc
        result = {
            "question_id": question.question_id,
            "question": question.text,
            "gold_answer": question.answer,
            "prediction": prediction,
            "group": question.group,
            "question_date": question.question_date,
            "metadata": question.metadata,
            **scores,
        }
        write_json(cache_path, result)
        if logger is not None:
            logger.log_agent_event(
                "reasoner",
                "qa_done",
                log_path=trace_path,
                question_id=question.question_id,
            )
        return result


def build_memorizer_prompt(
    *,
    window: dict[str, Any],
    memory_focus: str,
) -> str:
    start = int(window["start_line"])
    end = int(window["end_line"])
    return (
        "Background memory-ingestion tick. Do not answer an evaluation question.\n"
        "This workspace keeps one persistent memory state. A chronological batch of new "
        "sessions was just appended to trajectory; process only its new physical line "
        "window. Session boundaries are represented by changes in the time field.\n\n"
        f"New trajectory lines: L{start}-L{end}\n\n"
        "Required reads before shell commands:\n"
        f"1. read_file data/trajectory/trajectory.jsonl offset={start - 1} "
        f"limit={end - start + 1}\n"
        "2. read_file memory/Memories.md; continue if it does not exist yet.\n"
        "3. read_file skills/memorizer-memory/SKILL.md before using the skill CLI.\n\n"
        f"{memory_focus}\n\n"
        "Run one initial inspect: python skills/memorizer-memory/scripts/memory.py inspect "
        "--workspace .\n"
        "Run it exactly as written, without cd, Set-Location, drive paths, or navigation "
        "commands. Stage and commit only durable memories supported by this session window. "
        "Run all writes sequentially, never in parallel. To reduce tool round trips, group "
        "up to eight independent stage commands in one shell call separated by command "
        "semicolons; inspect and commit are ordered barriers. A second inspect is allowed "
        "only after an episode commit when its assigned IDs are needed for Semantic evidence. "
        "PowerShell data arguments such as --content must use single quotes; encode an "
        "apostrophe inside them as two apostrophes. Never double-quote data containing $. "
        "Use semicolons between multiple trajectory evidence atoms. Finish with one line "
        "starting MEMORY_COMMITTED: or NO_DURABLE_MEMORY:."
    )


def normalize_speaker(role: str) -> str:
    return "User" if role.lower() in {"user", "human"} else "Agent"


def load_models(
    *,
    model_name: str | None = None,
    judge_model_name: str | None = None,
) -> tuple[Any, Any]:
    config = load_config()
    agent_config = config.model_copy(update={"model": model_name}) if model_name else config
    judge_config = (
        config.model_copy(update={"model": judge_model_name})
        if judge_model_name
        else agent_config
    )
    model = build_chat_model(agent_config)
    judge_model = build_chat_model(judge_config)
    if model is None or judge_model is None:
        raise RuntimeError("MEMORIZER_MODEL and model credentials must be configured.")
    return model, judge_model


def load_agent_model(*, model_name: str | None = None) -> Any:
    config = load_config()
    if model_name:
        config = config.model_copy(update={"model": model_name})
    model = build_chat_model(config)
    if model is None:
        raise RuntimeError("MEMORIZER_MODEL and model credentials must be configured.")
    return model


def f1_score(prediction: str, gold: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1_score(prediction: str, gold: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens:
        return float(not gold_tokens)
    if not gold_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    precision = overlap / len(prediction_tokens)
    if precision == 0.0:
        return 0.0
    brevity_penalty = (
        1.0
        if len(prediction_tokens) >= len(gold_tokens)
        else math.exp(1.0 - len(gold_tokens) / len(prediction_tokens))
    )
    return brevity_penalty * precision


def normalize_answer(value: str) -> str:
    lowered = value.lower()
    no_punctuation = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def sample_fingerprint(sample: Sample) -> str:
    payload = json.dumps(asdict(sample), ensure_ascii=False, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def last_assistant_content(response: dict[str, Any]) -> str:
    for message in reversed(response.get("messages", [])):
        if isinstance(message, dict):
            if message.get("role") == "assistant":
                return message_content(message.get("content", ""))
        elif getattr(message, "type", None) == "ai":
            return message_content(getattr(message, "content", ""))
    return ""


def message_content(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        return message_content(message.get("content", ""))
    if isinstance(message, list):
        parts = []
        for item in message:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    content = getattr(message, "content", None)
    return message_content(content) if content is not None else str(message)


def parse_json_object(raw: str) -> dict[str, Any]:
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            raise ValueError(f"response does not contain JSON: {raw[:500]}") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def memory_completion_marker(value: str) -> str | None:
    match = re.search(
        r"(?m)^\s*(?:\*\*)?(MEMORY_COMMITTED:|NO_DURABLE_MEMORY:)",
        value,
    )
    return match.group(1) if match else None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return (normalized or sha256(value.encode("utf-8")).hexdigest()[:12])[:120]
