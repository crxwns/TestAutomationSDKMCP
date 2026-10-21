import asyncio
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from mcp.client._memory import InMemoryTransport
from mcp.client.session import ClientSession
from numpy.typing import NDArray

from test_automation_sdk_mcp.build_index import build_lexical_index
from test_automation_sdk_mcp.config import RuntimeConfig
from test_automation_sdk_mcp.errors import ArtifactError
from test_automation_sdk_mcp.index import write_document_store
from test_automation_sdk_mcp.lexical import (
    SQLITE_FILENAME,
    LexicalArtifacts,
    load_lexical_artifacts,
    load_runtime_artifacts,
    write_lexical_index,
)
from test_automation_sdk_mcp.server import AdaptiveRetriever, create_server

from .test_server import FailingEmbeddingProvider, FakeEmbeddingProvider, make_artifacts


def lexical_generation(tmp_path: Path) -> Path:
    directory = tmp_path / "lexical"
    directory.mkdir()
    store = make_artifacts(tmp_path).documents
    path = directory / "TA_Docu.documents.json"
    write_document_store(path, store)
    write_lexical_index(directory / SQLITE_FILENAME, path, store)
    return directory


def test_build_from_existing_chunks_needs_neither_raw_docs_nor_model(tmp_path: Path) -> None:
    source = lexical_generation(tmp_path)
    destination = tmp_path / "output"
    assert build_lexical_index(source / "TA_Docu.documents.json", destination, from_documents=True) == 2
    assert isinstance(load_runtime_artifacts(destination, vectors=True), LexicalArtifacts)
    assert sorted(path.name for path in destination.iterdir()) == ["TA_Docu.documents.json", SQLITE_FILENAME]


@pytest.mark.parametrize("vector_state", ["missing", "corrupt"])
def test_lexical_generation_survives_missing_or_corrupt_vector_artifacts(tmp_path: Path, vector_state: str) -> None:
    directory = lexical_generation(tmp_path)
    if vector_state == "corrupt":
        (directory / "TA_Docu.faiss").write_bytes(b"broken")
        (directory / "TA_Docu.manifest.json").write_text("not json")
    artifacts = load_runtime_artifacts(directory, vectors=True)
    retriever = AdaptiveRetriever(artifacts, None, RuntimeConfig())
    try:
        results = asyncio.run(retriever.retrieve("first"))
        assert [result.chunk_id for result in results] == ["first"]
        assert results[0].score_type == "bm25"
    finally:
        retriever.close()


def test_sqlite_rejects_mixed_or_corrupt_generations(tmp_path: Path) -> None:
    directory = lexical_generation(tmp_path)
    path = directory / "TA_Docu.documents.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ArtifactError, match="generation"):
        load_lexical_artifacts(directory)
    (directory / SQLITE_FILENAME).write_bytes(b"invalid sqlite")
    with pytest.raises(ArtifactError, match="invalid"):
        load_lexical_artifacts(directory)


def test_sqlite_rejects_modified_fts_rows(tmp_path: Path) -> None:
    directory = lexical_generation(tmp_path)
    with sqlite3.connect(directory / SQLITE_FILENAME) as connection:
        connection.execute("UPDATE chunks SET content = 'tampered' WHERE id = 'first'")
    with pytest.raises(ArtifactError, match="chunks"):
        load_lexical_artifacts(directory)


def test_auto_uses_vector_success_and_lexical_fallback_with_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = make_artifacts(tmp_path)
    provider = FakeEmbeddingProvider()
    config = RuntimeConfig(embedding_cooldown=10)
    retriever = AdaptiveRetriever(artifacts, provider, config)
    assert asyncio.run(retriever.retrieve("first"))[0].search_method == "vector"
    retriever.close()

    class CountingFailure(FailingEmbeddingProvider):
        calls = 0

        async def embed(self, inputs: Sequence[str]) -> NDArray[np.float32]:
            self.calls += 1
            return await super().embed(inputs)

    failing = CountingFailure()
    clock = [100.0]
    monkeypatch.setattr("test_automation_sdk_mcp.server.monotonic", lambda: clock[0])
    retriever = AdaptiveRetriever(artifacts, failing, config)
    try:
        assert asyncio.run(retriever.retrieve("first"))[0].search_method == "lexical"
        assert asyncio.run(retriever.retrieve("first"))[0].chunk_id == "first"
        assert failing.calls == 1
        clock[0] += 11
        asyncio.run(retriever.retrieve("first"))
        assert failing.calls == 2
    finally:
        retriever.close()


def test_hybrid_fuses_by_id_and_keeps_lexical_only_hits(tmp_path: Path) -> None:
    # Vector ranks first before second; lexical query only matches second.
    retriever = AdaptiveRetriever(
        make_artifacts(tmp_path), FakeEmbeddingProvider(), RuntimeConfig(search_mode="hybrid")
    )
    try:
        results = asyncio.run(retriever.retrieve("second"))
        assert [result.chunk_id for result in results] == ["second", "first"]
        assert results[0].distance == pytest.approx(1 / 61 + 1 / 62)
        assert all(result.score_type == "rrf" for result in results)
    finally:
        retriever.close()


def test_lexical_treats_query_as_text_and_never_calls_provider(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider()
    retriever = AdaptiveRetriever(make_artifacts(tmp_path), provider, RuntimeConfig(search_mode="lexical", model=None))
    try:
        assert asyncio.run(retriever.retrieve('first OR " * - :'))[0].chunk_id == "first"
        assert asyncio.run(retriever.retrieve('" * - :')) == []
        assert provider.inputs == []
    finally:
        retriever.close()


def test_no_endpoint_mcp_session_retrieves_without_constructing_provider(tmp_path: Path) -> None:
    directory = lexical_generation(tmp_path)
    config = RuntimeConfig.from_environment({"TA_SDK_DB_DIR": str(directory)})
    assert not config.embeddings_enabled

    def forbidden_provider(_: RuntimeConfig):
        raise AssertionError("must not construct an embedding provider")

    server = create_server(config, provider_factory=forbidden_provider)

    async def exercise() -> None:
        async with (
            InMemoryTransport(server) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool("retrieve_documentation", {"query": "first"})
            assert not result.is_error
            assert result.structured_content is not None
            snippets = result.structured_content["result"]
            assert snippets[0]["score_type"] == "bm25"
            assert snippets[0]["chunk_id"] == "first"
            assert "distance" not in snippets[0]

    asyncio.run(exercise())


def test_builder_cli_creates_lexical_generation_without_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from test_automation_sdk_mcp.build_index import main

    for key in list(os.environ):
        if key.startswith("TA_SDK_"):
            monkeypatch.delenv(key)
    source = lexical_generation(tmp_path)
    output = tmp_path / "new-generation"
    assert main(["--from-documents", str(source / "TA_Docu.documents.json"), "--output", str(output)]) == 0
    assert len(load_lexical_artifacts(output).documents.documents) == 2


def test_legacy_chunks_remain_usable_without_faiss(tmp_path: Path) -> None:
    from test_automation_sdk_mcp.chunking import hash_file
    from test_automation_sdk_mcp.index import write_index_manifest

    artifacts = make_artifacts(tmp_path)
    path = tmp_path / "TA_Docu.documents.json"
    write_document_store(path, artifacts.documents)
    manifest = artifacts.manifest.model_copy(update={"documents_sha256": hash_file(path)})
    write_index_manifest(tmp_path / "TA_Docu.manifest.json", manifest)
    assert isinstance(load_runtime_artifacts(tmp_path, vectors=True), LexicalArtifacts)


def test_lexical_rebuild_removes_stale_vector_artifacts(tmp_path: Path) -> None:
    directory = lexical_generation(tmp_path)
    (directory / "TA_Docu.faiss").write_bytes(b"stale vectors")
    (directory / "TA_Docu.manifest.json").write_text("stale manifest")
    assert build_lexical_index(directory / "TA_Docu.documents.json", directory, from_documents=True) == 2
    assert not (directory / "TA_Docu.faiss").exists()
    assert not (directory / "TA_Docu.manifest.json").exists()
    assert len(load_lexical_artifacts(directory).documents.documents) == 2


def test_builder_falls_back_when_configured_embeddings_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_automation_sdk_mcp import build_index as builder
    from test_automation_sdk_mcp.errors import EmbeddingError

    from .test_build_index import make_source

    source = make_source(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setenv("TA_SDK_OLLAMA_URL", "http://127.0.0.1:11434")

    async def fail(source: Path, output: Path) -> builder.BuildResult:
        raise EmbeddingError("model unavailable")

    monkeypatch.setattr(builder, "_build_from_arguments", fail)
    assert builder.main(["--source", str(source), "--output", str(output)]) == 0
    assert len(load_lexical_artifacts(output).documents.documents) == 2
