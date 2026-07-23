from __future__ import annotations

from datetime import datetime
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
import argparse
import json
import re
from collections import defaultdict

from eval.common import (
    EvaluationHarness,
    JudgeResult,
    Message,
    Question,
    Sample,
    Session,
    message_content,
    parse_json_object,
    load_models,
    write_json,
)
from eval.longmemeval import prompts


QUESTION_TYPES = {
    "temporal-reasoning",
    "multi-session",
    "knowledge-update",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
}


def question_prompt(question: Question) -> str:
    return prompts.QA_PROMPT.format(
        question_date=question.question_date or "(not specified)",
        question_type=question.group,
        question=question.text,
    )


def judge_question(model: Any, question: Question, prediction: str) -> dict[str, Any]:
    if not hasattr(model, "invoke"):
        raise TypeError(
            "LongMemEval LLM-as-judge requires an invokable chat model. Configure an "
            "OpenAI-compatible MEMORIZER_MODEL/OPENAI_BASE_URL."
        )
    response = model.invoke(
        [
            {
                "role": "system",
                "content": "You are a strict evaluation judge. Return valid JSON only.",
            },
            {
                "role": "user",
                "content": prompts.JUDGE_PROMPT.format(
                    question_type=question.group,
                    question_date=question.question_date or "(not specified)",
                    question=question.text,
                    gold=question.answer,
                    prediction=prediction,
                ),
            },
        ]
    )
    raw = message_content(response).strip()
    parsed = parse_json_object(raw)
    label = str(parsed.get("label", "")).upper()
    if label not in {"CORRECT", "WRONG"}:
        raise ValueError(f"judge returned an unsupported label: {raw[:500]}")
    return {
        "judge": asdict(
            JudgeResult(
                label=label,
                score=1.0 if label == "CORRECT" else 0.0,
                reasoning=str(parsed.get("reasoning", "")).strip(),
                raw=raw,
            )
        )
    }


def summarize_longmemeval(results: list[dict[str, Any]]) -> dict[str, Any]:
    questions = [question for result in results for question in result["questions"]]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        groups[str(question.get("group", "unknown"))].append(question)

    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        correct = sum(
            1 for item in items if item["judge"]["label"] == "CORRECT"
        )
        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
        }

    return {
        "benchmark": "longmemeval",
        "sample_count": len(results),
        "overall": metrics(questions),
        "by_question_type": {
            name: metrics(items) for name, items in sorted(groups.items())
        },
    }


def load_longmemeval(path: Path, question_types: set[str]) -> list[Sample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("samples", raw.get("questions")))
    if not isinstance(raw, list):
        raise ValueError("LongMemEval dataset must be a JSON list")

    samples = []
    for sample_index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"LongMemEval sample {sample_index} is not an object")
        question_type = str(item.get("question_type", "unknown"))
        if question_type not in question_types:
            continue

        question_id = str(
            item.get("question_id")
            or item.get("id")
            or f"longmemeval_{sample_index}"
        )
        sessions = parse_sessions(item)
        question = str(item.get("question", item.get("query", ""))).strip()
        if not question:
            continue
        samples.append(
            Sample(
                sample_id=question_id,
                sessions=sessions,
                questions=[
                    Question(
                        question_id=question_id,
                        text=question,
                        answer=answer_text(
                            item.get("answer", item.get("gold_answer", ""))
                        ),
                        group=question_type,
                        question_date=string_or_none(item.get("question_date")),
                        metadata={
                            "answer_session_ids": item.get("answer_session_ids", []),
                        },
                    )
                ],
                metadata={
                    "benchmark": "longmemeval",
                    "sample_index": sample_index,
                    "question_type": question_type,
                },
            )
        )
    return samples


def parse_sessions(item: dict[str, Any]) -> list[Session]:
    raw_sessions = item.get("haystack_sessions", item.get("sessions", []))
    dates = item.get("haystack_dates", item.get("session_dates", []))
    session_ids = item.get("haystack_session_ids", item.get("session_ids", []))
    if not isinstance(raw_sessions, list):
        raise ValueError("LongMemEval haystack_sessions must be a list")

    sessions = []
    for session_index, raw_session in enumerate(raw_sessions):
        raw_messages = normalize_session_messages(raw_session)
        session_id = (
            str(session_ids[session_index])
            if isinstance(session_ids, list) and session_index < len(session_ids)
            else f"session_{session_index}"
        )
        date = (
            str(dates[session_index])
            if isinstance(dates, list) and session_index < len(dates)
            else None
        )
        messages = [
            message
            for message in (
                parse_message(raw_message, message_index)
                for message_index, raw_message in enumerate(raw_messages)
            )
            if message is not None
        ]
        sessions.append(Session(session_id=session_id, date=date, messages=messages))
    return sorted(sessions, key=session_sort_key)


def session_sort_key(session: Session) -> tuple[int, Any]:
    if session.date:
        cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", session.date).strip()
        try:
            return (0, datetime.strptime(cleaned, "%Y/%m/%d %H:%M"))
        except ValueError:
            pass
    return (1, session.session_id)


def normalize_session_messages(raw_session: Any) -> list[Any]:
    if isinstance(raw_session, list):
        return raw_session
    if isinstance(raw_session, dict):
        messages = raw_session.get("messages", raw_session.get("conversation", []))
        return messages if isinstance(messages, list) else []
    return []


def parse_message(raw: Any, message_index: int) -> Message | None:
    if isinstance(raw, str):
        role = "user" if message_index % 2 == 0 else "assistant"
        return Message(speaker=role.title(), role=role, content=raw)
    if not isinstance(raw, dict):
        return None

    role = str(raw.get("role", raw.get("speaker", "unknown")))
    speaker = str(raw.get("name") or raw.get("speaker") or role.title())
    content = raw.get("content", raw.get("text", raw.get("message", "")))
    if isinstance(content, list):
        content = "\n".join(
            str(part.get("text", ""))
            if isinstance(part, dict)
            else str(part)
            for part in content
        )
    content = str(content).strip()
    if not content:
        return None
    return Message(speaker=speaker, role=role, content=content)


def answer_text(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def string_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ARM on LongMemEval after full-haystack memory construction."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to longmemeval_s_cleaned.json",
    )
    parser.add_argument("--run-name", required=True, help="Stable name for this evaluation run")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional agent model override; defaults to MEMORIZER_MODEL",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional metric judge model override; defaults to --model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help="Parent directory for evaluation artifacts",
    )
    parser.add_argument(
        "--question-types",
        default=",".join(sorted(QUESTION_TYPES)),
        help="Comma-separated LongMemEval question types",
    )
    parser.add_argument(
        "--question-ids",
        default=None,
        help="Optional comma-separated question IDs",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument(
        "--session-batch-size",
        type=int,
        default=1,
        help="Chronological sessions processed per Memorizer tick (default: 1)",
    )
    parser.add_argument(
        "--qa-workers",
        type=int,
        default=1,
        help="Questions answered concurrently after memory is complete (default: 1)",
    )
    parser.add_argument(
        "--sample-workers",
        type=int,
        default=1,
        help="Independent samples evaluated concurrently (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume sample memory construction and skip completed sample results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question_types = {
        value.strip() for value in args.question_types.split(",") if value.strip()
    }
    selected_ids = (
        {value.strip() for value in args.question_ids.split(",") if value.strip()}
        if args.question_ids
        else None
    )
    samples = load_longmemeval(args.dataset.resolve(), question_types)
    if selected_ids is not None:
        samples = [sample for sample in samples if sample.sample_id in selected_ids]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("No LongMemEval samples matched the requested filters.")
    if args.sample_workers < 1:
        raise SystemExit("--sample-workers must be at least 1")

    model, judge_model = load_models(
        model_name=args.model,
        judge_model_name=args.judge_model,
    )
    run_root = (args.output_dir / "longmemeval" / args.run_name).resolve()
    harness = EvaluationHarness(
        run_root=run_root,
        benchmark="longmemeval",
        model=model,
        memory_focus=prompts.MEMORY_FOCUS,
        question_prompt=question_prompt,
        score_question=lambda question, prediction: judge_question(
            judge_model, question, prediction
        ),
        resume=args.resume,
        session_batch_size=args.session_batch_size,
        qa_workers=args.qa_workers,
    )

    def evaluate_indexed(item: tuple[int, Sample]) -> dict[str, Any]:
        index, sample = item
        print(f"[LongMemEval {index}/{len(samples)}] {sample.sample_id}", flush=True)
        return harness.evaluate_sample(sample)

    indexed_samples = list(enumerate(samples, start=1))
    if args.sample_workers == 1:
        results = [evaluate_indexed(item) for item in indexed_samples]
    else:
        with ThreadPoolExecutor(max_workers=args.sample_workers) as executor:
            results = list(executor.map(evaluate_indexed, indexed_samples))

    summary = summarize_longmemeval(results)
    summary["run_name"] = args.run_name
    summary["dataset"] = str(args.dataset.resolve())
    summary["question_types"] = sorted(question_types)
    summary["sample_workers"] = args.sample_workers
    write_json(run_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
