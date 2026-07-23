from memorizer_agent.paths import WorkspacePaths
from memorizer_agent.runtime.consumption import latest_snapshot_version, write_consumption


class MemoryContextInjector:
    def __init__(self, paths: WorkspacePaths, max_chars: int = 12000) -> None:
        self.paths = paths
        self.max_chars = max_chars

    def build_context_message(self) -> str:
        latest_diff = self.paths.diffs / "latest.diff.md"
        if latest_diff.exists():
            diff_content = latest_diff.read_text(encoding="utf-8")
            if diff_content.strip():
                return self._format_message("Latest memory diff", diff_content)

        if self.paths.memories.exists():
            return "Memory file is available at ./memory/Memories.md. Read it yourself before answering prior-fact questions."

        return "No memory context is available yet. If needed, check whether ./memory/Memories.md exists."

    def mark_consumed_latest(self) -> None:
        write_consumption(self.paths.consumption, latest_snapshot_version(self.paths.snapshots))

    def _format_message(self, title: str, content: str) -> str:
        return f"{title}:\n{self._truncate(content)}"

    def _truncate(self, content: str) -> str:
        if len(content) <= self.max_chars:
            return content

        marker = "\n[Memory context truncated]\n"
        return content[: max(0, self.max_chars - len(marker))] + marker
