import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import pytest

import test_automation_sdk_mcp.build_index as build_index_module
from test_automation_sdk_mcp.build_index import BuildResult, build_index
from test_automation_sdk_mcp.config import DEFAULT_ARTIFACT_DIRECTORY, EmbeddingProviderKind, RuntimeConfig
from test_automation_sdk_mcp.documents import ChunkingManifest, IndexManifest
from test_automation_sdk_mcp.errors import IndexBuildError
from test_automation_sdk_mcp.index import artifact_paths, load_verified_artifacts
from test_automation_sdk_mcp.provider import EmbeddingProvenance


class TrackingProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, inputs: Sequence[str]) -> np.ndarray:
        self.calls.append(tuple(inputs))
        return np.ones((len(inputs), 768), dtype=np.float32)

    async def aclose(self) -> None:
        return None


class CapturingProvider:
    instances: ClassVar[list["CapturingProvider"]] = []

    def __init__(
        self,
        endpoint_url: str,
        model: str,
        *,
        api_key: str | None,
        connect_timeout: float,
        request_timeout: float,
    ) -> None:
        self.arguments = (endpoint_url, model, api_key, connect_timeout, request_timeout)
        self.close_calls = 0
        self.__class__.instances.append(self)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None

    async def aclose(self) -> None:
        self.close_calls += 1


class FailingProvider:
    async def embed(self, inputs: Sequence[str]) -> np.ndarray:
        raise RuntimeError("simulated embedding failure")

    async def aclose(self) -> None:
        return None


def make_source(root: Path) -> Path:
    source = root / "data"
    source.mkdir()
    (source / "index.html").write_text("raw html", encoding="utf-8")
    items: list[dict[str, object]] = [
        {
            "location": "index.html#one",
            "level": 1,
            "title": "One",
            "text": "<p>First</p>",
            "path": [],
            "tags": [],
        },
        {
            "location": "index.html#two",
            "level": 1,
            "title": "Two",
            "text": "<p>Second</p>",
            "path": [],
            "tags": [],
        },
    ]
    (source / "search.json").write_text(json.dumps({"items": items}), encoding="utf-8")
    return source


def artifact_bytes(output: Path) -> dict[str, bytes]:
    paths = artifact_paths(output)
    return {
        path.name: path.read_bytes()
        for path in (paths.faiss, paths.documents, paths.manifest, output / "TA_Docu.sqlite3")
    }


def make_result(output: Path, model: str = "fake") -> BuildResult:
    digest = "a" * 64
    manifest = IndexManifest(
        schema_version=2,
        index_type="IndexFlatL2",
        distance_metric="l2",
        embedding_provider="ollama",
        embedding_model=model,
        embedding_dimension=768,
        document_count=2,
        search_json_sha256=digest,
        html_tree_sha256=digest,
        faiss_sha256=digest,
        documents_sha256=digest,
        chunking=ChunkingManifest(max_characters=1000, overlap_characters=200),
    )
    return BuildResult(manifest, 2, 2, output)


def provenance(model: str) -> EmbeddingProvenance:
    return EmbeddingProvenance(EmbeddingProviderKind.OLLAMA, model)


def test_tiny_build_batches_and_publishes_verified_artifacts(tmp_path: Path) -> None:
    provider = TrackingProvider()
    output = tmp_path / "db"

    result = asyncio.run(build_index(make_source(tmp_path), output, provider, provenance("fake"), batch_size=1))
    artifacts = load_verified_artifacts(output)

    assert result.source_sections == 2
    assert result.chunk_count == 2
    assert [len(call) for call in provider.calls] == [1, 1]
    assert artifacts.index.ntotal == len(artifacts.documents.documents) == 2
    assert artifacts.manifest.embedding_dimension == 768
    assert artifacts.manifest.embedding_provider == "ollama"
    assert artifacts.manifest.embedding_model == "fake"


def test_builder_requires_explicit_provenance_descriptor(tmp_path: Path) -> None:
    with pytest.raises(IndexBuildError, match="explicit immutable"):
        asyncio.run(
            build_index(
                tmp_path / "missing-source",
                tmp_path / "missing-output",
                TrackingProvider(),
                object(),  # type: ignore[arg-type]
            )
        )


def test_cli_build_reads_runtime_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TA_SDK_OLLAMA_URL", "https://configured.example.test///")
    monkeypatch.setenv("TA_SDK_OLLAMA_MODEL", "configured-model")
    monkeypatch.setenv("TA_SDK_OLLAMA_API_KEY", "secret")
    monkeypatch.setenv("TA_SDK_CONNECT_TIMEOUT", "7.5")
    monkeypatch.setenv("TA_SDK_REQUEST_TIMEOUT", "42")
    captured = CapturingProvider.instances
    captured.clear()
    expected_manifest = IndexManifest(
        schema_version=2,
        index_type="IndexFlatL2",
        distance_metric="l2",
        embedding_provider="ollama",
        embedding_model="configured-model",
        embedding_dimension=768,
        document_count=0,
        search_json_sha256="a" * 64,
        html_tree_sha256="a" * 64,
        faiss_sha256="a" * 64,
        documents_sha256="a" * 64,
        chunking=ChunkingManifest(max_characters=1000, overlap_characters=200),
    )
    expected_result = BuildResult(expected_manifest, 0, 0, tmp_path)

    async def fake_build_index(
        source_directory: Path,
        output_directory: Path,
        provider: object,
        provenance: EmbeddingProvenance,
        **_: object,
    ) -> BuildResult:
        assert (source_directory, output_directory, provider) == (Path("source"), tmp_path, captured[0])
        assert provenance == EmbeddingProvenance(EmbeddingProviderKind.OLLAMA, "configured-model")
        return expected_result

    def fake_provider_factory(config: RuntimeConfig) -> CapturingProvider:
        assert config.model is not None
        return CapturingProvider(
            config.endpoint_url,
            config.model,
            api_key=config.api_key,
            connect_timeout=config.connect_timeout,
            request_timeout=config.request_timeout,
        )

    monkeypatch.setattr(build_index_module, "create_embedding_provider", fake_provider_factory)
    monkeypatch.setattr(build_index_module, "build_index", fake_build_index)

    result = asyncio.run(build_index_module._build_from_arguments(Path("source"), tmp_path))  # pyright: ignore[reportPrivateUsage]

    assert result == expected_result
    assert captured[0].arguments == (
        "https://configured.example.test",
        "configured-model",
        "secret",
        7.5,
        42.0,
    )
    assert captured[0].close_calls == 1


def test_cli_parser_supports_defaults_and_overrides(tmp_path: Path) -> None:
    defaults = build_index_module._parser().parse_args([])  # pyright: ignore[reportPrivateUsage]
    overrides = build_index_module._parser().parse_args(["--source", "custom-data", "--output", str(tmp_path)])  # pyright: ignore[reportPrivateUsage]

    assert defaults.source == Path("data")
    assert defaults.output == DEFAULT_ARTIFACT_DIRECTORY
    assert overrides.source == Path("custom-data")
    assert overrides.output == tmp_path


def test_cli_success_prints_summary_and_progress(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    async def fake_build_from_arguments(source: Path, output: Path) -> BuildResult:
        assert source == Path("custom-data")
        assert output == tmp_path
        return make_result(output, model="configured-model")

    monkeypatch.setenv("TA_SDK_OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(build_index_module, "_build_from_arguments", fake_build_from_arguments)

    result = build_index_module.main(["--source", "custom-data", "--output", str(tmp_path)])
    captured = capsys.readouterr()

    assert result == 0
    assert "Building documentation index from custom-data..." in captured.err
    assert "model=configured-model" in captured.out
    assert "source_sections=2" in captured.out
    assert "chunks=2 vectors=2 dimensions=768" in captured.out
    assert f"output={tmp_path}" in captured.out


def test_cli_failure_is_safe_and_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def failing_build_from_arguments(source: Path, output: Path) -> BuildResult:
        raise IndexBuildError("configured source is unavailable")

    monkeypatch.setenv("TA_SDK_OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(build_index_module, "_build_from_arguments", failing_build_from_arguments)

    result = build_index_module.main(["--source", "missing", "--output", "out"])
    captured = capsys.readouterr()

    assert result == 1
    assert "error: configured source is unavailable" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_embedding_failure_preserves_existing_artifacts(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    output = tmp_path / "db"
    asyncio.run(build_index(source, output, TrackingProvider(), provenance("first")))
    before = artifact_bytes(output)

    with pytest.raises(IndexBuildError):
        asyncio.run(build_index(source, output, FailingProvider(), provenance("second")))

    assert artifact_bytes(output) == before


def test_staging_failure_preserves_existing_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = make_source(tmp_path)
    output = tmp_path / "db"
    asyncio.run(build_index(source, output, TrackingProvider(), provenance("first")))
    before = artifact_bytes(output)

    def fail_staging(path: Path, index: object) -> None:
        raise OSError("simulated staging failure")

    monkeypatch.setattr(build_index_module, "_write_faiss", fail_staging)
    with pytest.raises(IndexBuildError):
        asyncio.run(build_index(source, output, TrackingProvider(), provenance("second")))

    assert artifact_bytes(output) == before


def test_mid_publication_failure_rolls_back_all_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = make_source(tmp_path)
    output = tmp_path / "db"
    asyncio.run(build_index(source, output, TrackingProvider(), provenance("first")))
    before = artifact_bytes(output)
    original_replace = os.replace
    replace_count = 0

    def fail_mid_publication(source_path: str | os.PathLike[str], destination_path: str | os.PathLike[str]) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 5:
            raise OSError("simulated mid-publication failure")
        original_replace(source_path, destination_path)

    monkeypatch.setattr(build_index_module.os, "replace", fail_mid_publication)
    with pytest.raises(IndexBuildError):
        asyncio.run(build_index(source, output, TrackingProvider(), provenance("second")))

    assert replace_count == 9
    assert artifact_bytes(output) == before
