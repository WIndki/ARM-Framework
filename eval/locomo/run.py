from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import argparse
import json
import re
from collections import defaultdict

from eval.common import (
    EvaluationHarness,
    Message,
    Question,
    Sample,
    Session,
    load_agent_model,
    bleu1_score,
    f1_score,
    write_json,
)
from eval.locomo import prompts


CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
SESSION_KEY = re.compile(r"^session_(\d+)$")


def question_prompt(question: Question) -> str:
    return prompts.QA_PROMPT.format(
        question_date=question.question_date or "(latest conversation session)",
        question=question.text,
    )


def score_question(question: Question, prediction: str) -> dict[str, Any]:
    return {
        "f1": f1_score(prediction, question.answer),
        "bleu1": bleu1_score(prediction, question.answer),
    }


def summarize_locomo(results: list[dict[str, Any]]) -> dict[str, Any]:
    questions = [question for result in results for question in result["questions"]]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        groups[str(question.get("group", "unknown"))].append(question)

    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        return {
            "total": total,
            "mean_f1": sum(item["f1"] for item in items) / total if total else 0.0,
            "mean_bleu1": (
                sum(item["bleu1"] for item in items) / total if total else 0.0
            ),
        }

    return {
        "benchmark": "locomo",
        "sample_count": len(results),
        "overall": metrics(questions),
        "by_category": {name: metrics(items) for name, items in sorted(groups.items())},
    }


def load_locomo(path: Path, categories: set[int]) -> list[Sample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("samples", raw.get("conversations")))
    if not isinstance(raw, list):
        raise ValueError("LoCoMo dataset must be a JSON list")

    samples = []
    for sample_index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"LoCoMo sample {sample_index} is not an object")
        sample_id = str(
            item.get("sample_id")
            or item.get("conversation_id")
            or item.get("id")
            or f"locomo_{sample_index}"
        )
        conversation = item.get("conversation")
        if not isinstance(conversation, dict):
            raise ValueError(f"LoCoMo sample {sample_id} has no conversation object")

        sessions = parse_sessions(conversation)
        reference_date = sessions[-1].date if sessions else None
        questions = parse_questions(item, sample_id, categories, reference_date)
        if questions:
            samples.append(
                Sample(
                    sample_id=sample_id,
                    sessions=sessions,
                    questions=questions,
                    metadata={
                        "benchmark": "locomo",
                        "sample_index": sample_index,
                        "speaker_a": conversation.get("speaker_a"),
                        "speaker_b": conversation.get("speaker_b"),
                    },
                )
            )
    return samples


def parse_sessions(conversation: dict[str, Any]) -> list[Session]:
    indexed_keys: list[tuple[tuple[int, Any], str]] = []
    for key in conversation:
        match = SESSION_KEY.fullmatch(key)
        if match:
            date = string_or_none(
                conversation.get(f"{key}_date_time")
                or conversation.get(f"{key}_date")
            )
            parsed_date = parse_locomo_date(date)
            sort_key = (
                (0, parsed_date)
                if parsed_date is not None
                else (1, int(match.group(1)))
            )
            indexed_keys.append((sort_key, key))

    sessions = []
    speaker_a = string_or_none(conversation.get("speaker_a"))
    for _, key in sorted(indexed_keys):
        raw_messages = conversation.get(key)
        if not isinstance(raw_messages, list):
            continue
        messages = [
            message
            for message in (
                parse_message(raw_message, message_index, speaker_a)
                for message_index, raw_message in enumerate(raw_messages)
            )
            if message is not None
        ]
        sessions.append(
            Session(
                session_id=key,
                date=string_or_none(
                    conversation.get(f"{key}_date_time")
                    or conversation.get(f"{key}_date")
                ),
                messages=messages,
            )
        )
    return sessions


def parse_message(
    raw: Any,
    message_index: int,
    speaker_a: str | None,
) -> Message | None:
    if isinstance(raw, str):
        return Message(
            speaker=f"speaker_{message_index % 2}",
            role="conversation",
            content=raw,
        )
    if not isinstance(raw, dict):
        return None

    speaker = str(raw.get("speaker") or raw.get("name") or f"speaker_{message_index % 2}")
    content = first_text(raw, "text", "content", "utterance", "message")
    query = first_text(raw, "query")
    caption = first_text(raw, "blip_caption", "caption", "image_caption")
    if query and caption:
        image_text = f"[Shared image; query: {query}; image content: {caption}]"
    elif query:
        image_text = f"[Shared image; query: {query}]"
    elif caption:
        image_text = f"[Shared image content: {caption}]"
    else:
        image_text = ""
    if image_text:
        content = f"{content}\n{image_text}" if content else image_text
    if not content:
        return None
    role = "user" if speaker_a is not None and speaker == speaker_a else "assistant"
    return Message(speaker=speaker, role=role, content=content)


def parse_questions(
    item: dict[str, Any],
    sample_id: str,
    categories: set[int],
    reference_date: str | None,
) -> list[Question]:
    raw_questions = item.get("qa", item.get("qa_pairs", []))
    if not isinstance(raw_questions, list):
        raise ValueError(f"LoCoMo sample {sample_id} has an invalid qa field")

    questions = []
    for question_index, raw in enumerate(raw_questions):
        if not isinstance(raw, dict):
            continue
        category = int(raw.get("category", 0))
        if category not in categories:
            continue
        question_text = first_text(raw, "question", "query")
        answer = answer_text(raw.get("answer", raw.get("gold_answer", "")))
        if not question_text:
            continue
        questions.append(
            Question(
                question_id=str(
                    raw.get("question_id")
                    or raw.get("id")
                    or f"{sample_id}:q{question_index}"
                ),
                text=question_text,
                answer=answer,
                group=CATEGORY_NAMES.get(category, f"category-{category}"),
                question_date=reference_date,
                metadata={
                    "category": category,
                    "evidence": raw.get("evidence"),
                },
            )
        )
    return questions


def parse_locomo_date(value: str | None) -> datetime | None:
    if value is None:
        return None
    for date_format in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(value, date_format)
        except ValueError:
            continue
    return None


def first_text(value: dict[str, Any], *keys: str) -> str:
    for key in keys:
        candidate = value.get(key)
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return ""


def answer_text(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def string_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ARM on LoCoMo after full-sample memory construction."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to locomo10.json")
    parser.add_argument("--run-name", required=True, help="Stable name for this evaluation run")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional agent model override; defaults to MEMORIZER_MODEL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help="Parent directory for evaluation artifacts",
    )
    parser.add_argument(
        "--categories",
        default="1,2,3,4",
        help="Comma-separated LoCoMo categories; category 5 is excluded by default",
    )
    parser.add_argument(
        "--sample-ids",
        default=None,
        help="Optional comma-separated sample IDs",
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
        "--resume",
        action="store_true",
        help="Resume sample memory construction and skip completed sample results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = {int(value) for value in args.categories.split(",") if value.strip()}
    selected_ids = (
        {value.strip() for value in args.sample_ids.split(",") if value.strip()}
        if args.sample_ids
        else None
    )
    samples = load_locomo(args.dataset.resolve(), categories)
    if selected_ids is not None:
        samples = [sample for sample in samples if sample.sample_id in selected_ids]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("No LoCoMo samples matched the requested filters.")

    model = load_agent_model(model_name=args.model)
    run_root = (args.output_dir / "locomo" / args.run_name).resolve()
    harness = EvaluationHarness(
        run_root=run_root,
        benchmark="locomo",
        model=model,
        memory_focus=prompts.MEMORY_FOCUS,
        question_prompt=question_prompt,
        score_question=score_question,
        resume=args.resume,
        session_batch_size=args.session_batch_size,
        qa_workers=args.qa_workers,
    )

    results = []
    for index, sample in enumerate(samples, start=1):
        print(f"[LoCoMo {index}/{len(samples)}] {sample.sample_id}", flush=True)
        results.append(harness.evaluate_sample(sample))

    summary = summarize_locomo(results)
    summary["run_name"] = args.run_name
    summary["dataset"] = str(args.dataset.resolve())
    summary["categories"] = sorted(categories)
    write_json(run_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
