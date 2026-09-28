from __future__ import annotations

import os
from dataclasses import dataclass

from collector.constants import DEFAULT_MAX_WORKERS


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    text = value.strip()
    return text or None


@dataclass(frozen=True)
class Settings:
    database_url: str | None = None
    openalex_mailto: str | None = None
    openalex_api_key: str | None = None
    max_workers: int = DEFAULT_MAX_WORKERS
    request_timeout_s: float = 30.0
    user_agent: str = "weak-signal-collector/0.1 (research; document-collector)"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_url=_env("DATABASE_URL"),
            openalex_mailto=_env("COLLECTOR_OPENALEX_MAILTO"),
            # OPEN_ALEX — второе имя того же ключа, оно уже лежит в .env у команды.
            openalex_api_key=_env("OPENALEX_API_KEY") or _env("OPEN_ALEX"),
            max_workers=int(_env("COLLECTOR_MAX_WORKERS") or DEFAULT_MAX_WORKERS),
            request_timeout_s=float(_env("COLLECTOR_REQUEST_TIMEOUT_S") or 30.0),
            user_agent=_env("COLLECTOR_USER_AGENT")
            or "weak-signal-collector/0.1 (research; document-collector)",
        )
