# ARM: Asynchronous Reasoner–Memorizer Framework

ARM is a reasoning–memory decoupling framework for long-horizon agents. It separates task-time reasoning from background memory maintenance while preserving deterministic, traceable memory transitions.

## Method

### Reasoning–Memory Decoupling

ARM has three components:

- **Reasoner** solves the current task from working context and relevant memory.
- **Memorizer** asynchronously compresses new interactions into long-term memory.
- **Harness** records trajectories, verifies candidate memories, commits valid updates, and synchronizes memory deltas.

The Reasoner keeps a bounded task context. Memory maintenance runs independently instead of repeatedly placing the full interaction history in the Reasoner's context.

### Hierarchical Memory

At step `t`, ARM maintains:

```text
M_t = (Trajectory_t, Episodic_t, Semantic_t, Working_t)
```

| Level | Content | Access |
| --- | --- | --- |
| Trajectory | Chronological observations and actions | Reasoner, Memorizer |
| Episodic | Concrete events with trajectory provenance | Reasoner, Memorizer |
| Semantic | Stable facts, status, and provenance | Reasoner, Memorizer |
| Working | Task history, synchronized deltas, retrieved entries | Reasoner |

Episodic memory is append-only. Semantic entries cite supporting episodes and trajectories and carry a status: `active`, `superseded`, `stale`, `contradicted`, or `downgraded`.

### Runtime Modules

**Incremental Compression** processes only trajectories accumulated since the Memorizer's previous update. It proposes new Episodic/Semantic entries or a Semantic status revision without reprocessing the full history.

**Provenance Verification** is deterministic. The Harness checks structure, resolves every direct and indirect evidence reference, rejects modifications to append-only Episodic memory, and atomically commits only valid candidate updates. Invalid candidates are discarded and returned for regeneration.

**Delta Synchronization** compares the current Semantic state with the last synchronized version and sends only changed Semantic entries to the Reasoner. Episodic additions are not included in the delta; the Reasoner can retrieve them directly when needed.

### Asynchronous Feedback Loop

At each task step:

1. The Reasoner receives working memory plus the latest verified Semantic delta and produces an action.
2. The Harness appends the interaction to the trajectory.
3. In the background, the Memorizer compresses newly accumulated trajectory entries.
4. The Harness verifies and commits valid memory candidates.
5. The resulting Semantic delta becomes available to subsequent Reasoner steps.

```text
Reasoner -> interaction -> Harness -> trajectory
    ^                                  |
    |          Semantic delta          v
    +----------- Harness <--------- Memorizer
```

## Repository Mapping

```text
src/memorizer_agent/agents/       Reasoner, Memorizer, and runtime sessions
src/memorizer_agent/runtime/      trajectory, context injection, observability
skills/memorizer-memory/          deterministic memory CLI and verification gate
eval/locomo/                      LoCoMo experiment
eval/longmemeval/                 LongMemEval experiment
```

Trajectory is physical JSONL with only `id`, `speaker`, `time`, and `content`:

```json
{"id":"r00000001","speaker":"Alice","time":"2024-05-01 09:00","content":"Message text."}
```

Memory entries retain source provenance:

```text
[E-0001] Alice completed the hike. [Evidence: trajectory.jsonl:L9]
[S-0001 active] Alice owns a gray cat. [Evidence: Memories.md:E-0002@L5; trajectory.jsonl:L3]
```

## Setup

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
```

Create `.env` in the repository root:

```dotenv
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.example.com/v1
MEMORIZER_MODEL=openai:model-name
MEMORIZER_WORKSPACE=.
```

`OPENAI_BASE_URL` is optional for the default OpenAI endpoint. Lowercase aliases (`api_key`, `base_url`, `model`, `workspace`) are accepted.

### Interactive Runtime

```powershell
uv run arm --workspace .
uv run arm --workspace . show-memories
uv run arm --workspace . doctor
```

Without `MEMORIZER_MODEL`, the CLI uses placeholder agents for local harness checks.

## Experiments

### LoCoMo

Metrics: token F1 and BLEU-1.

```powershell
uv run python -m eval.locomo.run `
  --dataset H:\data\locomo10.json `
  --run-name locomo-run `
  --categories 1,2,3,4 `
  --session-batch-size 1 `
  --qa-workers 8
```

LoCoMo options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--categories` | `1,2,3,4` | Included question categories |
| `--sample-ids` | all | Comma-separated sample IDs |
| `--qa-workers` | `1` | Concurrent questions after memory freeze |

### LongMemEval

Metric: LLM-as-judge accuracy.

```powershell
uv run python -m eval.longmemeval.run `
  --dataset H:\data\longmemeval_s_cleaned.json `
  --run-name long-run `
  --session-batch-size 1 `
  --sample-workers 4
```

LongMemEval options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--question-types` | all supported | Included question types |
| `--question-ids` | all | Comma-separated question IDs |
| `--judge-model` | agent model | Judge model override |
| `--sample-workers` | `1` | Concurrent independent samples |
| `--qa-workers` | `1` | Concurrent questions within a sample |

### Shared Options

| Option | Default | Purpose |
| --- | --- | --- |
| `--dataset` | required | Dataset JSON path |
| `--run-name` | required | Stable output identifier |
| `--model` | `MEMORIZER_MODEL` | Agent model override |
| `--output-dir` | `eval/results` | Result root |
| `--session-batch-size` | `1` | Chronological sessions per Memorizer update |
| `--limit` | all | Maximum samples |
| `--resume` | off | Continue memory/QA checkpoints and skip completed samples |

### Outputs

```text
eval/results/<benchmark>/<run>/
|-- workspaces/<sample>/
|   |-- data/trajectory/trajectory.jsonl
|   `-- memory/Memories.md
|-- samples/<sample>/
|   |-- qa-0001-<question-id>.json
|   `-- sample.json
`-- summary.json
```

Logs are disabled by default. Set `DEBUG=true` to write per-question `.log.jsonl` files and workspace logs under `runtime/logs/`.
