MEMORY_FOCUS = """# Assistant Memory Goal

Preserve every non-filler detail that could support a later answer, with exact speaker attribution. Information supplied by any speaker is first-class evidence even when nobody repeats it.

## Coverage

- Scan every new line; skip only greetings, pure reactions, and exact repetition.
- Retain facts, answers, recommendations, commitments, generated details, preferences and anti-preferences, people, places, products, numbers, dates, reasons, plans, completed actions, and updates.
- Preserve all list members and values needed for later counting, comparison, duration, ordering, or calculation.
- Resolve adjacent references and explicit follow-up answers before writing.

## Representation

- Keep completed actions distinct from plans and proposals.
- Group adjacent facts only when they concern one event/topic; never merge unrelated contexts.
- Put one-off events in Episodic memory. Add Semantic memory only for stable current facts, preferences, ownership, relationships, or state.
- Keep time only when it changes meaning, order, duration, recency, a deadline/plan, an update, or repeated-event identity.
- When a newer fact changes current state, append it and revise the older Semantic status; never erase history or store session IDs."""

QA_PROMPT = """# Goal

Answer from the completed memory state. Think silently and return only the final answer.

## 1. Retrieve

- Read all relevant memory. Search exact entities, roles, variants, rare nouns, synonyms, and related events.
- If memory lacks a required detail or conflicts, inspect focused evidence in `data/trajectory/trajectory.jsonl`.
- Keep different people, products, and contexts separate. Information from every speaker may be the answer.

## 2. Resolve

- Current state: prefer the newest supported active belief; keep historical facts at their original time.
- Counts/order: enumerate distinct completed instances, deduplicate references, then answer.
- Calculations/comparisons: collect every required value before computing; never invent missing inputs.
- Actions: distinguish completed events from intentions, offers, and abandoned plans.
- Recommendations: respect strong preferences, anti-preferences, existing resources, and requested activity.

## 3. Time

- Question date: {question_date}
- Use source time for time-bounded questions and correct event disambiguation. Use the question date only when the question is relative to it.
- Compute durations from the correct endpoints; do not replace event-to-event time with today's date.

## 4. Sufficiency

- Make only direct, evidence-supported deductions.
- Before abstaining, search memory and trajectory for the entity and close synonyms.
- If required information is genuinely absent, answer exactly `The information provided is not enough`. For `Did I ever ...?`, no matching event means `No`.

## 5. Emit

- Return one concise plain-text answer that fills the requested slot.
- No reasoning, Markdown, citation, source discussion, evaluation language, or question restatement.

Question type: {question_type}
Question: {question}"""

JUDGE_PROMPT = """# Role

Judge whether the response is semantically correct. Return JSON only.

## Decision Rules

- Accept faithful paraphrases, equivalent date/unit formats, reasonable rounding, and extra supported detail.
- For compound answers, require every essential independent concept; optional examples need not match.
- Reject contradictions, wrong entities/contexts, missing core requirements, outdated current values, and unsupported answers.
- For personalized answers, require alignment with the stated preferences and anti-preferences; wording need not match the rubric.
- If the reference says information is insufficient, accept any clear abstention. If the reference is answerable, reject abstention.
- Use no outside knowledge and do not independently replace the reference answer.

## Item

- Question type: {question_type}
- Question date: {question_date}
- Question: {question}
- Reference: {gold}
- Response: {prediction}

Output exactly one JSON object:
`{{"label":"CORRECT|WRONG","reasoning":"one concise sentence"}}`"""
