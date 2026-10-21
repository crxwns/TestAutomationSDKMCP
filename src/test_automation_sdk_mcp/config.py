"""Explicit runtime configuration parsing and validation."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit

from .errors import ConfigurationError

DEFAULT_ENDPOINT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "nomic-embed-text:v1.5"
DEFAULT_RESULT_COUNT = 5
MAX_RESULT_COUNT = 50
DEFAULT_ARTIFACT_DIRECTORY = Path(__file__).parent / "db"
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_REQUEST_TIMEOUT = 30.0
MAX_TIMEOUT = 300.0


class EmbeddingProviderKind(str, Enum):
    """Supported embedding provider kinds."""

    OLLAMA = "ollama"
    OPENAI = "openai"


def _validate_endpoint(value: object) -> str:
    if not isinstance(value, str):
        raise ConfigurationError("endpoint_url must be a URL string.")

    endpoint = value.strip().rstrip("/")
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as error:
        raise ConfigurationError("endpoint_url must be a valid HTTP or HTTPS URL.") from error

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is None
        and ":" in parsed.netloc.rsplit("@", 1)[-1]
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigurationError("endpoint_url must be a valid HTTP or HTTPS URL.")
    return endpoint


def _validate_model(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 for char in value):
        raise ConfigurationError("model must be a non-empty model name.")
    return value.strip()


def _parse_provider(value: object) -> EmbeddingProviderKind:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("embedding provider must be ollama or openai.")
    try:
        return EmbeddingProviderKind(value.strip().lower())
    except ValueError as error:
        raise ConfigurationError("embedding provider must be ollama or openai.") from error


def _validate_api_key(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigurationError("api_key must be a string or None.")
    return value or None


def _validate_result_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_RESULT_COUNT:
        raise ConfigurationError(f"result_count must be an integer between 1 and {MAX_RESULT_COUNT}.")
    return value


def _validate_timeout(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a finite number between 0 and {MAX_TIMEOUT}.")
    timeout = float(value)
    if not 0 < timeout <= MAX_TIMEOUT or not isfinite(timeout):
        raise ConfigurationError(f"{name} must be a finite number between 0 and {MAX_TIMEOUT}.")
    return timeout


def _validate_artifact_directory(value: object) -> Path:
    raw_path: str
    if isinstance(value, str):
        raw_path = value
    elif isinstance(value, Path):
        raw_path = str(value)
    else:
        raise ConfigurationError("artifact_directory must be a valid path.")
    if not raw_path or not raw_path.strip() or "\x00" in raw_path:
        raise ConfigurationError("artifact_directory must be a valid path.")
    try:
        return Path(raw_path)
    except (OSError, TypeError, ValueError) as error:
        raise ConfigurationError("artifact_directory must be a valid path.") from error


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Validated values required by the runtime and artifact loader."""

    endpoint_url: str = DEFAULT_ENDPOINT_URL
    model: str | None = DEFAULT_MODEL
    api_key: str | None = field(default=None, repr=False)
    result_count: int = DEFAULT_RESULT_COUNT
    artifact_directory: Path = field(default_factory=lambda: DEFAULT_ARTIFACT_DIRECTORY)
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    provider: EmbeddingProviderKind = EmbeddingProviderKind.OLLAMA
    search_mode: str = "auto"
    embeddings_enabled: bool = True
    embedding_cooldown: float = 60.0

    def __post_init__(self) -> None:
        if self.search_mode not in {"auto", "hybrid", "lexical"}:
            raise ConfigurationError("search_mode must be auto, hybrid, or lexical.")
        object.__setattr__(self, "embedding_cooldown", _validate_timeout(self.embedding_cooldown, "embedding_cooldown"))
        provider = _parse_provider(self.provider)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "endpoint_url", _validate_endpoint(self.endpoint_url))
        if provider is EmbeddingProviderKind.OLLAMA and self.search_mode != "lexical":
            if self.model is None:
                raise ConfigurationError("Ollama embedding model is required.")
            object.__setattr__(self, "model", _validate_model(self.model))
        elif self.model is not None:
            object.__setattr__(self, "model", _validate_model(self.model))
        if (
            provider is EmbeddingProviderKind.OPENAI
            and urlsplit(self.endpoint_url).path.rstrip("/") != "/v1/embeddings"
        ):
            raise ConfigurationError("OpenAI endpoint_url must include /v1/embeddings.")
        object.__setattr__(self, "api_key", _validate_api_key(self.api_key))
        object.__setattr__(self, "result_count", _validate_result_count(self.result_count))
        object.__setattr__(self, "artifact_directory", _validate_artifact_directory(self.artifact_directory))
        object.__setattr__(self, "connect_timeout", _validate_timeout(self.connect_timeout, "connect_timeout"))
        object.__setattr__(self, "request_timeout", _validate_timeout(self.request_timeout, "request_timeout"))

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "RuntimeConfig":
        """Create configuration from environment variables at an explicit call site."""

        return runtime_config_from_environment(environ)


def _parse_int(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer.") from error


def _parse_float(value: str, name: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number.") from error


def runtime_config_from_environment(environ: Mapping[str, str] | None = None) -> RuntimeConfig:
    """Parse `TA_SDK_*` variables without reading the environment at import time."""

    values = os.environ if environ is None else environ
    search_mode = values.get("TA_SDK_SEARCH_MODE", "auto").strip().lower()
    provider = (
        _parse_provider(values.get("TA_SDK_EMBEDDING_PROVIDER", EmbeddingProviderKind.OLLAMA.value))
        if search_mode != "lexical"
        else EmbeddingProviderKind.OLLAMA
    )
    endpoint_variable = "TA_SDK_OLLAMA_URL" if provider is EmbeddingProviderKind.OLLAMA else "TA_SDK_OPENAI_URL"
    enabled = search_mode != "lexical" and bool(values.get(endpoint_variable))
    result_count = values.get("TA_SDK_RESULT_COUNT")
    connect_timeout = values.get("TA_SDK_CONNECT_TIMEOUT")
    request_timeout = values.get("TA_SDK_REQUEST_TIMEOUT")
    artifact_directory = values.get("TA_SDK_DB_DIR")
    if provider is EmbeddingProviderKind.OLLAMA:
        endpoint_url = (
            values.get("TA_SDK_OLLAMA_URL", DEFAULT_ENDPOINT_URL) if search_mode != "lexical" else DEFAULT_ENDPOINT_URL
        )
        model = values.get("TA_SDK_OLLAMA_MODEL", DEFAULT_MODEL) if search_mode != "lexical" else None
        api_key = values.get("TA_SDK_OLLAMA_API_KEY") or None
    else:
        endpoint_url = values.get("TA_SDK_OPENAI_URL")
        if endpoint_url is None or not endpoint_url.strip():
            raise ConfigurationError("TA_SDK_OPENAI_URL is required for the openai provider.")
        model_value = values.get("TA_SDK_OPENAI_MODEL")
        model = None if model_value is None else _validate_model(model_value)
        api_key = values.get("TA_SDK_OPENAI_API_KEY") or None
    return RuntimeConfig(
        endpoint_url=endpoint_url,
        model=model,
        api_key=api_key,
        result_count=DEFAULT_RESULT_COUNT if result_count is None else _parse_int(result_count, "result_count"),
        artifact_directory=DEFAULT_ARTIFACT_DIRECTORY if artifact_directory is None else Path(artifact_directory),
        connect_timeout=(
            DEFAULT_CONNECT_TIMEOUT if connect_timeout is None else _parse_float(connect_timeout, "connect_timeout")
        ),
        request_timeout=(
            DEFAULT_REQUEST_TIMEOUT if request_timeout is None else _parse_float(request_timeout, "request_timeout")
        ),
        provider=provider,
        search_mode=search_mode,
        embeddings_enabled=enabled,
        embedding_cooldown=_parse_float(values.get("TA_SDK_EMBEDDING_COOLDOWN", "60"), "embedding_cooldown"),
    )


load_runtime_config = runtime_config_from_environment
