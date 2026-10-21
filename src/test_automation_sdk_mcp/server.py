"""Read-only documentation retrieval and MCP server boundary."""

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from time import monotonic
from typing import Annotated, Literal, cast

import numpy as np
from mcp.server import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.types import ToolAnnotations
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .config import DEFAULT_ARTIFACT_DIRECTORY, RuntimeConfig
from .documents import EMBEDDING_DIMENSION
from .errors import (
    ApplicationErrorCode,
    ConfigurationError,
    EmbeddingError,
    RetrievalError,
    TestAutomationSDKError,
    to_mcp_error,
)
from .index import LoadedArtifacts, load_verified_artifacts
from .lexical import LexicalArtifacts, LexicalSearch, load_runtime_artifacts
from .provider import EmbeddingProvider
from .provider.factory import create_embedding_provider

MAX_QUERY_LENGTH = 4_000

SERVER_INSTRUCTIONS = (
    "Use this server when a user needs authoritative Test Automation SDK information or is developing, reviewing, "
    "or debugging Python tests that use dspace.testautomation. Call retrieve_documentation before making "
    "SDK-specific claims or choosing SDK APIs and patterns. Relevant topics include test environments and ports, "
    "variable access, capturing and triggers, scenario setup and execution, synchronization, and captured-data "
    "analysis. Do not call it solely for generic Python or pytest questions, test-domain logic that does not depend "
    "on the SDK, or unrelated dSPACE products. Base SDK-specific guidance on the returned snippets and retain their "
    "source locations."
)

TOOL_DESCRIPTION = (
    "Retrieve authoritative, sourced Test Automation SDK documentation for SDK questions and SDK-based Python test "
    "development. Use for exact APIs and patterns involving TestEnvironmentAccess or the ta fixture, variables, "
    "ports, Capturing or MultiPortCapturing, rasters and triggers, Scenario parameterization and lifecycle, "
    "synchronization, or capture result analysis. Do not use for generic Python or pytest questions or unrelated "
    "dSPACE products. Verified documentation chunks are required. Search modes: auto uses embeddings when "
    "configured, with SQLite fallback; lexical uses SQLite only; hybrid fuses SQLite and vector ranks via RRF. "
    "No embedding endpoint is needed for lexical retrieval. Vector scores are uncalibrated L2 distances, "
    "lexical scores are SQLite BM25 (lower is better), and RRF scores are higher-is-better. "
    "Embedding calls use configured connect and request timeouts; cancellation propagates and the server "
    "does not retry internally. Failed embedding calls trigger a cooldown and lexical fallback. "
    "Errors: invalid_query, embedding_request_timed_out, embedding_service_unavailable, "
    "embedding_service_rejected_request, embedding_service_failure, embedding_response_invalid, "
    "retrieval_failed, tool_output_invalid, server_runtime_unavailable."
)


class DocumentationSnippet(BaseModel):
    """Strict structured output returned by the MCP retrieval tool."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    content: str
    location: str
    title: str
    breadcrumbs: list[str]
    chunk_id: str
    search_method: Literal["vector", "lexical", "hybrid"]
    score: float
    score_type: Literal["l2", "bm25", "rrf"]

    @field_validator("content", "location", "title")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("snippet text fields must not be empty")
        return value

    @field_validator("score")
    @classmethod
    def validate_distance(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("score must be finite")
        return value


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Internal immutable representation of one nearest documentation result."""

    content: str
    location: str
    title: str
    breadcrumbs: tuple[str, ...]
    distance: float
    chunk_id: str = ""
    search_method: Literal["vector", "lexical", "hybrid"] = "vector"
    score_type: Literal["l2", "bm25", "rrf"] = "l2"


@dataclass(frozen=True, slots=True)
class _ServerRuntime:
    retriever: "AdaptiveRetriever"


EmbeddingProviderFactory = Callable[[RuntimeConfig], EmbeddingProvider]


def _validate_query(query: object) -> str:
    if not isinstance(query, str):
        raise RetrievalError("Documentation query must be a string.", code=ApplicationErrorCode.INVALID_QUERY)
    if not query.strip():
        raise RetrievalError("Documentation query must not be empty.", code=ApplicationErrorCode.INVALID_QUERY)
    if len(query) > MAX_QUERY_LENGTH:
        raise RetrievalError(
            f"Documentation query must be at most {MAX_QUERY_LENGTH} characters.",
            code=ApplicationErrorCode.INVALID_QUERY,
        )
    return query


def _validated_query_vector(vectors: object) -> NDArray[np.float32]:
    try:
        matrix = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        finite = bool(np.isfinite(matrix).all())
    except (OverflowError, TypeError, ValueError) as error:
        raise RetrievalError(
            "Embedding provider returned an invalid query vector.",
            code=ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID,
        ) from error
    if matrix.shape != (1, EMBEDDING_DIMENSION) or not finite:
        raise RetrievalError(
            "Embedding provider returned an invalid query vector.",
            code=ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID,
        )
    return matrix


def _search_index(index: object, query_vector: NDArray[np.float32], result_count: int) -> tuple[object, object]:
    search = getattr(index, "search", None)
    if not callable(search):
        raise RetrievalError(
            "The documentation index cannot execute a search.", code=ApplicationErrorCode.RETRIEVAL_FAILED
        )
    try:
        result = search(query_vector, result_count)
    except (RuntimeError, TypeError, ValueError) as error:
        raise RetrievalError(
            "The documentation index search failed.", code=ApplicationErrorCode.RETRIEVAL_FAILED
        ) from error
    if not isinstance(result, tuple):
        raise RetrievalError(
            "The documentation index returned an invalid search result.", code=ApplicationErrorCode.RETRIEVAL_FAILED
        )
    result_items = cast(tuple[object, ...], result)
    if len(result_items) != 2:
        raise RetrievalError(
            "The documentation index returned an invalid search result.", code=ApplicationErrorCode.RETRIEVAL_FAILED
        )
    return result_items[0], result_items[1]


class DocumentationRetriever:
    """Embed queries and map validated FAISS rows to documentation records."""

    def __init__(self, artifacts: LoadedArtifacts, provider: EmbeddingProvider, result_count: int) -> None:
        if isinstance(result_count, bool) or result_count <= 0:
            raise ConfigurationError("result_count must be a positive integer.")
        self._artifacts = artifacts
        self._provider = provider
        self._result_count = result_count

    @classmethod
    def from_directory(
        cls,
        artifact_directory: Path,
        provider: EmbeddingProvider,
        result_count: int,
    ) -> "DocumentationRetriever":
        """Load and validate one artifact generation before serving queries."""

        artifacts = load_verified_artifacts(artifact_directory)
        return cls(artifacts, provider, result_count)

    async def retrieve(self, query: str) -> list[RetrievalResult]:
        """Return nearest-first results with FAISS L2 distances."""

        validated_query = _validate_query(query)
        documents = self._artifacts.documents.documents
        if not documents:
            return []

        index_count = getattr(self._artifacts.index, "ntotal", None)
        if isinstance(index_count, bool) or not isinstance(index_count, int) or index_count < 0:
            raise RetrievalError(
                "The documentation index has an invalid vector count.", code=ApplicationErrorCode.RETRIEVAL_FAILED
            )
        result_count = min(self._result_count, index_count, len(documents))
        if result_count == 0:
            return []

        vectors = await self._provider.embed([validated_query])
        query_vector = _validated_query_vector(vectors)

        try:
            raw_distances, raw_row_ids = await asyncio.to_thread(
                _search_index,
                self._artifacts.index,
                query_vector,
                result_count,
            )
        except TestAutomationSDKError:
            raise
        except (RuntimeError, TypeError, ValueError) as error:
            raise RetrievalError(
                "The documentation index search failed.", code=ApplicationErrorCode.RETRIEVAL_FAILED
            ) from error

        try:
            distances = np.asarray(raw_distances, dtype=np.float64)
            row_ids = np.asarray(raw_row_ids)
            finite_distances = bool(np.isfinite(distances).all())
        except (OverflowError, TypeError, ValueError) as error:
            raise RetrievalError(
                "The documentation index returned invalid search data.", code=ApplicationErrorCode.RETRIEVAL_FAILED
            ) from error
        if distances.shape != (1, result_count) or not finite_distances:
            raise RetrievalError(
                "The documentation index returned invalid distances.", code=ApplicationErrorCode.RETRIEVAL_FAILED
            )
        if row_ids.shape != (1, result_count) or not np.issubdtype(row_ids.dtype, np.integer):
            raise RetrievalError(
                "The documentation index returned invalid row IDs.", code=ApplicationErrorCode.RETRIEVAL_FAILED
            )

        for distance in distances[0]:
            if distance < 0:
                raise RetrievalError(
                    "The documentation index returned an invalid L2 distance.",
                    code=ApplicationErrorCode.RETRIEVAL_FAILED,
                )

        ordered_positions = sorted(range(result_count), key=lambda position: float(distances[0][position]))
        results: list[RetrievalResult] = []
        for position in ordered_positions:
            row_id = int(row_ids[0][position])
            if row_id < 0 or row_id >= len(documents):
                raise RetrievalError(
                    "The documentation index returned an invalid row ID.", code=ApplicationErrorCode.RETRIEVAL_FAILED
                )
            document = documents[row_id]
            results.append(
                RetrievalResult(
                    content=document.content,
                    location=document.location,
                    title=document.title,
                    breadcrumbs=document.breadcrumbs,
                    distance=float(distances[0][position]),
                    chunk_id=document.id,
                )
            )
        return results


class AdaptiveRetriever(DocumentationRetriever):
    """Use SQLite independently and isolate expected failures of optional semantics."""

    def __init__(
        self,
        artifacts: LoadedArtifacts | LexicalArtifacts,
        provider: EmbeddingProvider | None,
        config: RuntimeConfig,
    ) -> None:
        self._store = artifacts.documents
        self._lexical_search = LexicalSearch(self._store)
        self._config = config
        self._vector = (
            DocumentationRetriever(artifacts, provider, min(50, config.result_count * 4))
            if isinstance(artifacts, LoadedArtifacts) and provider is not None and config.search_mode != "lexical"
            else None
        )
        self._unavailable_until = 0.0

    def close(self) -> None:
        self._lexical_search.close()

    async def _lexical(self, query: str) -> list[RetrievalResult]:
        rows = await asyncio.to_thread(self._lexical_search.search, query, min(50, self._config.result_count * 4))
        documents = {doc.id: doc for doc in self._store.documents}
        return [
            RetrievalResult(doc.content, doc.location, doc.title, doc.breadcrumbs, score, chunk_id, "lexical", "bm25")
            for chunk_id, score in rows
            for doc in [documents[chunk_id]]
        ]

    async def _semantic(self, query: str) -> list[RetrievalResult] | None:
        if self._vector is None or monotonic() < self._unavailable_until:
            return None
        try:
            return await self._vector.retrieve(query)
        except (RetrievalError, EmbeddingError):
            logging.getLogger(__name__).warning(
                "Semantic search unavailable; using SQLite during the embedding cooldown."
            )
            self._unavailable_until = monotonic() + self._config.embedding_cooldown
            return None

    async def retrieve(self, query: str) -> list[RetrievalResult]:
        query = _validate_query(query)
        count = self._config.result_count
        if self._config.search_mode == "hybrid":
            lexical, semantic = await asyncio.gather(self._lexical(query), self._semantic(query))
            if semantic is None:
                return lexical[:count]
            scores: dict[str, float] = {}
            snippets: dict[str, RetrievalResult] = {}
            for ranking in (lexical, semantic):
                for rank, result in enumerate(ranking, 1):
                    scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (60 + rank)
                    snippets[result.chunk_id] = result
            return [
                RetrievalResult(r.content, r.location, r.title, r.breadcrumbs, scores[key], key, "hybrid", "rrf")
                for key in sorted(scores, key=lambda key: (-scores[key], key))[:count]
                for r in [snippets[key]]
            ]
        semantic = await self._semantic(query)
        return semantic[:count] if semantic is not None else (await self._lexical(query))[:count]


def _default_provider_factory(config: RuntimeConfig) -> EmbeddingProvider:
    return create_embedding_provider(config)


async def _close_provider(provider: EmbeddingProvider) -> None:
    await provider.aclose()


def _snippet(result: RetrievalResult) -> DocumentationSnippet:
    return DocumentationSnippet(
        content=result.content,
        location=result.location,
        title=result.title,
        breadcrumbs=list(result.breadcrumbs),
        chunk_id=result.chunk_id,
        search_method=result.search_method,
        score=result.distance,
        score_type=result.score_type,
    )


def create_server(
    config: RuntimeConfig | None = None,
    *,
    artifacts: LoadedArtifacts | None = None,
    provider: EmbeddingProvider | None = None,
    provider_factory: EmbeddingProviderFactory | None = None,
) -> MCPServer[_ServerRuntime]:
    """Create the single-tool MCP server with verified, immutable artifacts."""

    runtime_config = RuntimeConfig.from_environment() if config is None else config
    if provider is not None and provider_factory is not None:
        raise ConfigurationError("provider and provider_factory cannot both be configured.")
    if artifacts is not None:
        loaded_artifacts = artifacts
    elif runtime_config.artifact_directory == DEFAULT_ARTIFACT_DIRECTORY:
        loaded_artifacts = load_runtime_artifacts(
            DEFAULT_ARTIFACT_DIRECTORY,
            vectors=runtime_config.embeddings_enabled and runtime_config.search_mode != "lexical",
        )
    else:
        loaded_artifacts = load_runtime_artifacts(
            runtime_config.artifact_directory,
            vectors=runtime_config.embeddings_enabled and runtime_config.search_mode != "lexical",
        )
    factory = _default_provider_factory if provider_factory is None else provider_factory
    direct_retriever = (
        None
        if provider is None
        else AdaptiveRetriever(
            loaded_artifacts,
            provider,
            runtime_config,
        )
    )
    active_runtime: _ServerRuntime | None = None

    @asynccontextmanager
    async def lifespan(_: MCPServer[_ServerRuntime]) -> AsyncGenerator[_ServerRuntime, None]:
        nonlocal active_runtime
        owned_provider = provider
        if (
            owned_provider is None
            and isinstance(loaded_artifacts, LoadedArtifacts)
            and runtime_config.embeddings_enabled
            and runtime_config.search_mode != "lexical"
        ):
            owned_provider = factory(runtime_config)
        active_runtime = _ServerRuntime(
            retriever=direct_retriever
            or AdaptiveRetriever(
                loaded_artifacts,
                owned_provider,
                runtime_config,
            )
        )
        try:
            yield active_runtime
        finally:
            active_runtime.retriever.close()
            active_runtime = None
            if owned_provider is not None:
                await _close_provider(owned_provider)

    server = MCPServer[_ServerRuntime](
        name="Test Automation SDK",
        description="Authoritative documentation retrieval for the Test Automation SDK and SDK-based test development.",
        instructions=SERVER_INSTRUCTIONS,
        version=version("test-automation-sdk-mcp"),
        lifespan=lifespan,
    )

    async def retrieve_documentation(
        query: Annotated[
            str,
            Field(
                strict=True,
                json_schema_extra={"minLength": 1, "maxLength": MAX_QUERY_LENGTH, "pattern": r"\S"},
                description=(
                    "One self-contained Test Automation SDK question or test-development task, including relevant "
                    "API names, SDK concepts, desired behavior, or error details."
                ),
            ),
        ],
        context: Context[_ServerRuntime, object],
    ) -> list[DocumentationSnippet]:
        """Retrieve relevant documentation for one non-empty query."""

        try:
            if direct_retriever is not None:
                retriever = direct_retriever
            else:
                try:
                    runtime_context = context.request_context.lifespan_context
                    retriever = runtime_context.retriever
                except (AttributeError, ValueError) as error:
                    if active_runtime is None:
                        project_error = RetrievalError(
                            "The documentation server runtime is unavailable.",
                            code=ApplicationErrorCode.SERVER_RUNTIME_UNAVAILABLE,
                        )
                        raise to_mcp_error(project_error) from error
                    retriever = active_runtime.retriever
            results = await retriever.retrieve(query)
            try:
                return [_snippet(result) for result in results]
            except ValidationError as error:
                project_error = RetrievalError(
                    "The documentation server returned an invalid snippet.",
                    code=ApplicationErrorCode.TOOL_OUTPUT_INVALID,
                )
                raise to_mcp_error(project_error) from error
        except TestAutomationSDKError as error:
            if error.code is None:
                raise
            raise to_mcp_error(error) from error

    server.add_tool(
        retrieve_documentation,
        name="retrieve_documentation",
        description=TOOL_DESCRIPTION,
        annotations=ToolAnnotations.model_validate(
            {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": True,
            }
        ),
        structured_output=True,
    )

    return server


__all__ = [
    "MAX_QUERY_LENGTH",
    "SERVER_INSTRUCTIONS",
    "TOOL_DESCRIPTION",
    "DocumentationRetriever",
    "DocumentationSnippet",
    "EmbeddingProviderFactory",
    "RetrievalResult",
    "create_server",
]
