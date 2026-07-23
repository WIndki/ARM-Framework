from pathlib import Path
from typing import Any
import os

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class MemorizerConfig(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    workspace: Path


def _first_value(values: dict[str, str | None], *names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
        value = values.get(name)
        if value is not None:
            return value
    return None


def load_config() -> MemorizerConfig:
    cwd = Path.cwd()
    values = dotenv_values(cwd / ".env")
    workspace_value = _first_value(values, "MEMORIZER_WORKSPACE", "workspace") or "."
    workspace = Path(workspace_value)
    if not workspace.is_absolute():
        workspace = cwd / workspace

    return MemorizerConfig(
        api_key=_first_value(values, "OPENAI_API_KEY", "api_key"),
        base_url=_first_value(values, "OPENAI_BASE_URL", "base_url"),
        model=_first_value(values, "MEMORIZER_MODEL", "model"),
        workspace=workspace.resolve(),
    )


def apply_runtime_environment(config: MemorizerConfig) -> None:
    if config.api_key and "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = config.api_key
    if config.base_url and "OPENAI_BASE_URL" not in os.environ:
        os.environ["OPENAI_BASE_URL"] = config.base_url


def build_chat_model(config: MemorizerConfig) -> Any:
    if not config.model:
        return None

    provider, model_name = _split_model_name(config.model)
    if provider != "openai" and not config.base_url:
        return config.model

    return ChatOpenAI(
        model=model_name,
        api_key=config.api_key,
        base_url=config.base_url,
        use_responses_api=False,
    )


def _split_model_name(model: str) -> tuple[str | None, str]:
    if ":" not in model:
        return None, model
    provider, model_name = model.split(":", 1)
    return provider, model_name
