import json
from pathlib import Path
from typing import Annotated

import typer

from memorizer_agent.agents.sessions import (
    BackgroundMemorizerLoop,
    BackgroundMemorizerPlugin,
    EchoReasonerAgent,
    ReasonerSession,
    SilentMemorizerAgent,
    create_background_memorizer_plugin,
    create_reasoner_session,
)
from memorizer_agent.agents.factory import MEMORY_SKILL_SCRIPT
from memorizer_agent.config import apply_runtime_environment, build_chat_model, load_config
from memorizer_agent.paths import WorkspacePaths

app = typer.Typer(help="Asynchronous Reasoner-Memorizer framework CLI.")


def _paths(ctx: typer.Context) -> WorkspacePaths:
    workspace = ctx.obj.get("workspace") if ctx.obj else None
    if workspace is None:
        workspace = load_config().workspace
    return WorkspacePaths(Path(workspace).resolve())


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", help="Workspace root. Defaults to config."),
    ] = None,
) -> None:
    ctx.obj = {"workspace": workspace}
    if ctx.invoked_subcommand is None:
        config = load_config()
        paths = WorkspacePaths(Path(workspace or config.workspace).resolve())
        paths.ensure()
        _run_interactive_cli(paths, build_chat_model(config))


@app.command("show-memories")
def show_memories(ctx: typer.Context) -> None:
    paths = _paths(ctx)
    if paths.memories.exists():
        typer.echo(paths.memories.read_text(encoding="utf-8"), nl=False)
    else:
        typer.echo("")


@app.command()
def doctor(ctx: typer.Context) -> None:
    paths = _paths(ctx)
    paths.ensure()
    typer.echo(
        json.dumps(
            {
                "workspace": str(paths.root),
                "memory_skill_script": str((paths.root / MEMORY_SKILL_SCRIPT).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run_interactive_cli(paths: WorkspacePaths, model) -> None:
    if model:
        apply_runtime_environment(load_config())
        reasoner = create_reasoner_session(paths, model)
        memorizer = create_background_memorizer_plugin(paths, model)
    else:
        reasoner = ReasonerSession(paths=paths, agent=EchoReasonerAgent())
        memorizer = BackgroundMemorizerPlugin(paths=paths, agent=SilentMemorizerAgent())
    memorizer_loop = BackgroundMemorizerLoop(memorizer)
    typer.echo("ARM")
    memorizer_loop.start()
    try:
        while True:
            try:
                user_message = typer.prompt(">")
            except (EOFError, KeyboardInterrupt):
                typer.echo("")
                break
            if user_message.strip().lower() in {"exit", "quit"}:
                break
            response = reasoner.ask(user_message)
            typer.echo(f"Reasoner: {response}")
    finally:
        memorizer_loop.stop()
