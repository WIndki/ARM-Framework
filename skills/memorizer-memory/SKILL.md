---
name: memorizer-memory
description: Use when the background Memorizer needs to inspect, stage, revise, validate, or commit append-only memories from trajectory evidence.
---

# Memorizer Memory

Memory gate and harness live inside this skill. The foreground `memorizer-agent` app is not the memory gate. The background Memorizer is responsible for reading trajectory evidence and existing memories, then using this skill CLI to stage and commit durable append-only memory.

Only public script: `scripts/memory.py`.
Internal implementation: `scripts/gate.py`.

Skill CLI commands run from the workspace root. Run the skill script by relative path and pass `--workspace .` so the gate resolves durable paths from the anchored workspace. The shell examples below are for the `memory.py` CLI only; they are not examples for reading project files.

## Required Context Reads

Before any write command, use `read_file`, not shell readers, to read the evidence and state you need:

1. Read `data/trajectory/trajectory.jsonl` at the line window provided by the loop prompt. Use the tool's `offset` and `limit` arguments so you read only the relevant physical lines.
2. Read `memory/Memories.md` before deduplication, Semantic Memory decisions, or belief revisions. If it does not exist yet, continue; the first commit can create it.
3. Read this `SKILL.md` in a fresh Memorizer context before using the CLI. Keep this contract in context across later loop ticks. Reread it after context compression, command failure, or uncertainty.

Do not infer trajectory content from the loop prompt. The prompt provides line coordinates; the file read provides evidence.

Prefer `read_file` for Markdown, JSON, JSONL, ledger, snapshot, and diff content. Avoid `Get-Content`, `type`, `cat`, `more`, directory listings, and broad shell searches for file reading unless `read_file` failed and you are diagnosing that failure. Use workspace-relative read paths such as `data/trajectory/trajectory.jsonl`, `memory/Memories.md`, and `skills/memorizer-memory/SKILL.md`; do not convert them to Windows absolute paths.

## Commands

```powershell
python skills/memorizer-memory/scripts/memory.py inspect --workspace .
python skills/memorizer-memory/scripts/memory.py stage-episode --content '...' --evidence trajectory.jsonl:L12-L18 --workspace .
python skills/memorizer-memory/scripts/memory.py commit --workspace .
python skills/memorizer-memory/scripts/memory.py stage-semantic --content '...' --episode E-0001 --status active --trajectory-evidence trajectory.jsonl:L12-L18 --workspace .
python skills/memorizer-memory/scripts/memory.py revise-belief --memory-id S-0001 --status superseded_by:S-0002 --workspace .
python skills/memorizer-memory/scripts/memory.py clear-staging --workspace .
```

PowerShell data arguments such as `--content` must use single quotes so `$` remains literal. Inside a single-quoted value, write an apostrophe as two apostrophes: `'User''s budget is $2,500.'` Never use double quotes for data containing `$`.

`inspect` reports current version and pending staged records. It is a status check only. Do not stop after `inspect` when the new trajectory window contains durable information.

Version `0`, missing `memory/Memories.md`, missing ledger, missing snapshots, and missing diffs are normal for a fresh workspace. They are not reasons to return `NO_DURABLE_MEMORY`; the gate creates durable artifacts during `commit`.

## Workflow

1. Read trajectory and memory context with `read_file`.
2. Run the initial `inspect`; inspect once more only after an episode commit when newly assigned IDs are required for Semantic evidence.
3. Decide whether the new trajectory window has durable memory value.
4. Stage Episodic Memory from physical trajectory line offsets.
5. Commit episodes.
6. Stage Semantic Memory from committed episode IDs when a stable preference, requirement, project decision, correction, or belief update is supported.
7. Commit semantics and belief revisions.
8. If staging is wrong or validation fails, run `clear-staging`, then restage.

Return `NO_DURABLE_MEMORY` only when the new trajectory window has no durable memory value. Fresh storage is not a blocker.

## Selection Rules

Create Episodic memories for concrete durable events:

- user requirements, constraints, corrections, and approvals
- assistant deliverables or implementation decisions the user may ask about later
- important user feedback, preferences, or changes of direction
- specific story, research, or project facts likely to be referenced later
- non-redundant facts, answers, recommendations, commitments, or generated details stated by the Agent

Inspect each trajectory record's `speaker`, `time`, and `content`. Every speaker
is a valid memory source. Skip only exact repetition or genuinely low-value
phrasing, never a fact merely because of its speaker label.

Time is selective, not mandatory. Before staging, ask whether removing `time`
would change the event's meaning, order, duration, currentness, deadline,
update semantics, or identity among repeated events. If not, omit it. Timeless
preferences, owned tools, general facts, and standalone recommendations
normally omit time. Completed events, plans, state changes, and temporal
calculations often retain it. Never write internal session IDs into memory.
Never ignore a relevant time and never invent one.

Do not memorize routine greetings, filler, duplicates, transient wording, or every assistant sentence.

Episodic granularity should be narrow. If one trajectory window contains several meaningful requests, corrections, decisions, or deliverables, stage several Episodic records. Do not compress the whole window into one broad episode.

Semantic memories are for stable beliefs and durable summaries, not a replacement for Episodic evidence. Prefer Semantic Memory for user preferences, long-lived project rules, current decisions, and belief status.

## Format Rules

`Memories.md` has two sections only:

```md
## Episodic Memory
[E-0001] content [Evidence: trajectory.jsonl:L21-L22]

## Semantic Memory
[S-0001 active] content [Evidence: Memories.md:E-0001@L4; trajectory.jsonl:L21-L22]
```

Each memory is one line. Evidence must include concrete line offset coordinates:

- `trajectory.jsonl:L21`
- `trajectory.jsonl:L21-L22`
- `Memories.md:E-0001@L4`

Semantic memories must cite committed Episodic support. The gate computes the final `Memories.md:E-xxxx@Lx` offset.

The `stage-semantic --episode E-xxxx` argument must reference an episode that has already been committed. If you just staged episodes, commit them first, then use the committed episode IDs for Semantic Memory.

When citing trajectory evidence, use the narrowest supporting range. Do not use the whole loop window by default if one line or a smaller range supports the memory.

## Append-Only Rules

Do not delete, overwrite, renumber, or collapse memories. To change a belief, append a new Semantic Memory and revise the old status.

Allowed statuses:

- `active`
- `downgraded`
- `stale`
- `superseded_by:S-xxxx`
- `contradicted_by:S-xxxx`

## Do Not

- Do not call `memorizer-agent memory ...`; that command must not exist.
- Do not edit `memory/Memories.md`, ledger, snapshots, diffs, or staging files directly.
- Do not write durable memory with `write_file` or `edit_file`; durable writes go through `scripts/memory.py`.
- Do not create extra wrapper scripts when `memory.py` can express the operation.
- Do not invent evidence, IDs, or line offsets.
- Do not treat `inspect` as a complete memory update.
- Do not activate meta-harness proposals as code during background memory maintenance.
