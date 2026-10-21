from pathlib import Path

import pytest

from test_automation_sdk_mcp.config import (
    DEFAULT_ARTIFACT_DIRECTORY,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_ENDPOINT_URL,
    DEFAULT_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RESULT_COUNT,
    EmbeddingProviderKind,
    RuntimeConfig,
    runtime_config_from_environment,
)
from test_automation_sdk_mcp.errors import ConfigurationError


def test_default_configuration() -> None:
    config = runtime_config_from_environment({})

    assert config.endpoint_url == DEFAULT_ENDPOINT_URL
    assert config.model == DEFAULT_MODEL
    assert config.api_key is None
    assert config.result_count == DEFAULT_RESULT_COUNT
    assert config.artifact_directory == DEFAULT_ARTIFACT_DIRECTORY
    assert config.connect_timeout == DEFAULT_CONNECT_TIMEOUT
    assert config.request_timeout == DEFAULT_REQUEST_TIMEOUT


def test_environment_overrides_are_normalized_and_secret_is_not_represented() -> None:
    secret = "super-secret-token"
    config = RuntimeConfig.from_environment(
        {
            "TA_SDK_OLLAMA_URL": "https://embedding.example.test///",
            "TA_SDK_OLLAMA_MODEL": " custom-model ",
            "TA_SDK_OLLAMA_API_KEY": secret,
            "TA_SDK_RESULT_COUNT": "12",
            "TA_SDK_DB_DIR": "artifacts",
            "TA_SDK_CONNECT_TIMEOUT": "2.5",
            "TA_SDK_REQUEST_TIMEOUT": "45",
        }
    )

    assert config.endpoint_url == "https://embedding.example.test"
    assert config.model == "custom-model"
    assert config.api_key == secret
    assert config.result_count == 12
    assert config.artifact_directory == Path("artifacts")
    assert config.connect_timeout == 2.5
    assert config.request_timeout == 45.0
    assert secret not in repr(config)
    assert secret not in str(config)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RuntimeConfig(endpoint_url="not-a-url"),
        lambda: RuntimeConfig(endpoint_url="ftp://example.test"),
        lambda: RuntimeConfig(model="  "),
        lambda: RuntimeConfig(result_count=0),
        lambda: RuntimeConfig(result_count=51),
        lambda: RuntimeConfig(connect_timeout=0),
        lambda: RuntimeConfig(request_timeout=-1),
        lambda: runtime_config_from_environment({"TA_SDK_DB_DIR": "\x00invalid"}),
    ],
)
def test_invalid_configuration_values_are_rejected(factory: object) -> None:
    with pytest.raises(ConfigurationError):
        factory()  # type: ignore[operator]


def test_invalid_environment_integer_preserves_parse_cause_without_echoing_value() -> None:
    secret_like_value = "not-a-count-secret"

    with pytest.raises(ConfigurationError) as raised:
        runtime_config_from_environment({"TA_SDK_RESULT_COUNT": secret_like_value})

    assert isinstance(raised.value.__cause__, ValueError)
    assert secret_like_value not in str(raised.value)


def test_configuration_error_does_not_echo_api_key() -> None:
    secret = "super-secret-token"

    with pytest.raises(ConfigurationError) as raised:
        RuntimeConfig(endpoint_url="not-a-url", api_key=secret)

    assert secret not in str(raised.value)


def test_openai_configuration_selects_only_openai_values_and_supports_trust_mode() -> None:
    secret = "openai-secret-token"
    config = runtime_config_from_environment(
        {
            "TA_SDK_EMBEDDING_PROVIDER": " OPENAI ",
            "TA_SDK_OPENAI_URL": "https://embedding.example.test/v1/embeddings///",
            "TA_SDK_OPENAI_MODEL": " endpoint-model ",
            "TA_SDK_OPENAI_API_KEY": secret,
            "TA_SDK_OLLAMA_URL": "https://ignored.example.test",
            "TA_SDK_OLLAMA_MODEL": "ignored-model",
            "TA_SDK_OLLAMA_API_KEY": "ignored-secret",
        }
    )

    assert config.provider is EmbeddingProviderKind.OPENAI
    assert config.endpoint_url == "https://embedding.example.test/v1/embeddings"
    assert config.model == "endpoint-model"
    assert config.api_key == secret
    assert secret not in repr(config)

    trust_config = runtime_config_from_environment(
        {
            "TA_SDK_EMBEDDING_PROVIDER": "openai",
            "TA_SDK_OPENAI_URL": "http://127.0.0.1:8080/v1/embeddings",
        }
    )
    assert trust_config.model is None
    assert trust_config.api_key is None


@pytest.mark.parametrize(
    "environment",
    [
        {"TA_SDK_EMBEDDING_PROVIDER": ""},
        {"TA_SDK_EMBEDDING_PROVIDER": "unsupported"},
        {"TA_SDK_EMBEDDING_PROVIDER": "openai"},
        {"TA_SDK_EMBEDDING_PROVIDER": "openai", "TA_SDK_OPENAI_URL": "https://example.test"},
    ],
)
def test_invalid_openai_provider_configuration_is_rejected(environment: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError):
        runtime_config_from_environment(environment)


def test_search_defaults_to_lexical_without_explicit_endpoint() -> None:
    config = runtime_config_from_environment({})
    assert config.search_mode == "auto"
    assert not config.embeddings_enabled


def test_unselected_endpoint_does_not_enable_embeddings() -> None:
    config = runtime_config_from_environment({"TA_SDK_OPENAI_URL": "http://127.0.0.1:8080/v1/embeddings"})
    assert not config.embeddings_enabled


def test_lexical_ignores_embedding_settings() -> None:
    config = runtime_config_from_environment(
        {
            "TA_SDK_SEARCH_MODE": "LEXICAL",
            "TA_SDK_EMBEDDING_PROVIDER": "invalid",
            "TA_SDK_OLLAMA_URL": "invalid",
            "TA_SDK_OLLAMA_MODEL": "",
        }
    )
    assert config.search_mode == "lexical"
    assert not config.embeddings_enabled
    assert config.model is None


@pytest.mark.parametrize(
    "values",
    [
        {"TA_SDK_SEARCH_MODE": "unknown"},
        {"TA_SDK_EMBEDDING_COOLDOWN": "nan"},
        {"TA_SDK_EMBEDDING_COOLDOWN": "0"},
    ],
)
def test_invalid_search_configuration(values: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError):
        runtime_config_from_environment(values)
