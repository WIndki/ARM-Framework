MEMORY_FOCUS = """# Conversation Memory Goal

Preserve every non-filler detail that could answer a later question, with exact original-speaker attribution. Do not keep only major themes or impose a memory-count limit.

## Coverage

- Scan every new line. Skip only greetings, pure reactions, and exact repetition.
- Retain people, relationships, family, pets, preferences, hobbies, books/media, art, places, lists, counts, time, reasons, emotions, plans, outcomes, repeated activities, and useful image details.
- Preserve all list members and explicit answers. Resolve adjacent references such as `it`, `this`, or `yes, I made it` against the preceding turn.

## Representation

- A caption proves what an image depicts, not that a person did it; a speaker's explicit follow-up can prove ownership or action.
- Keep completed actions distinct from plans, offers, and attempts.
- Group adjacent facts only when they concern one event/topic; retain each answer-bearing fact and its evidence. Never merge unrelated topics.
- Put one-off events in Episodic memory. Add Semantic memory only for stable current facts, preferences, relationships, ownership, or state.
- Keep time only when it affects meaning, order, duration, recency, plans, updates, or event identity. Never store session IDs."""

QA_PROMPT = """# Goal

Answer the question from the completed conversation memory. Think silently; return only the final answer.

## 1. Retrieve

- Read all relevant memory, not only the first match. Search exact entities, rare nouns, synonyms, and related events.
- Use original speaker names for attribution.
- If memory lacks exact detail, is ambiguous, or may omit an item, run focused shell `rg` searches on `data/trajectory/trajectory.jsonl`. Include image captions and adjacent replies. Never guess a name, place, date, or number.
- Abstain only after focused searches of both memory and trajectory find no usable evidence.

## 2. Resolve

- Combine facts across sessions when they concern the same subject.
- Lists/counts: enumerate every supported distinct item or occurrence, deduplicate references to the same event, then answer. Different dates or settings normally mean different occurrences.
- Prefer the most specific supported fact. Distinguish completed actions from plans, offers, and untried ideas.
- Images show depicted content; adjacent dialogue may explicitly establish who made, owns, or used it.

## 3. Time

- Reference date: {question_date}
- Use source times, not today's date. A conversation time may differ from the event time.
- Durations require both endpoints. Similar events require date-based disambiguation.
- Preserve supported relative wording (`the week before ...`) instead of replacing it with a vague approximation. Use full month names.

## 4. Infer

- For hypothetical, opinion, or trait questions, make the shortest direct inference from stated causes, behavior, values, preferences, and symbolic actions.
- Preserve modality: use `likely yes/no` for hypothetical or predictive questions; do not refuse merely because the exact hypothetical was never stated.

## 5. Emit

- Fill only the grammatical answer slot. Reuse direct source wording and granularity when available.
- Factual yes/no: `yes` or `no`. Direct count: Arabic numeral. Duration: value plus unit. Requested title/name: only that title/name. Requested category/leaning: its conventional broad label.
- Remove unrequested authors, causes, locations, examples, related facts, evidence, and explanation.
- Output exactly one non-empty plain-text line. No label, Markdown, citation, rationale, hedging, restatement, or second answer.

Question: {question}"""
