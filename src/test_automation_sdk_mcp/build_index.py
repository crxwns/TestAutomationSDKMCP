"""Build and publish the documentation FAISS index."""

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from .chunking import (
    DEFAULT_MAX_CHARACTERS,
    DEFAULT_OVERLAP_CHARACTERS,
    DocumentChunk,
    chunk_items,
    hash_file,
    hash_html_tree,
    load_source_items,
)
from .config import DEFAULT_ARTIFACT_DIRECTORY, RuntimeConfig
from .documents import ChunkingManifest, DocumentRecord, DocumentStore, IndexManifest, load_document_store
from .errors import EmbeddingError, IndexBuildError, TestAutomationSDKError
from .index import (
    ArtifactPaths,
    load_verified_artifacts,
    write_document_store,
    write_index_manifest,
)
from .lexical import SQLITE_FILENAME, load_lexical_artifacts, write_lexical_index
from .provider import DEFAULT_BATCH_SIZE, EMBEDDING_DIMENSION, EmbeddingProvenance, EmbeddingProvider
from .provider.factory import create_embedding_provider


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Summary of one successfully published index generation."""

    manifest: IndexManifest
    source_sections: int
    chunk_count: int
    output_directory: Path


def _document_store(chunks: Sequence[DocumentChunk]) -> DocumentStore:
    return DocumentStore(
        schema_version=1,
        documents=tuple(
            DocumentRecord(
                id=chunk.id,
                location=chunk.location,
                title=chunk.title,
                breadcrumbs=chunk.breadcrumbs,
                tags=chunk.tags,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
            )
            for chunk in chunks
        ),
    )


def _validated_vectors(
    vectors: object,
    expected_count: int,
) -> NDArray[np.float32]:
    try:
        matrix = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    except (OverflowError, TypeError, ValueError) as error:
        raise EmbeddingError("Embedding provider returned invalid vectors.") from error
    if matrix.shape != (expected_count, EMBEDDING_DIMENSION) or not bool(np.isfinite(matrix).all()):
        raise EmbeddingError("Embedding provider returned invalid vectors.")
    return matrix


async def _embed_chunks(
    provider: EmbeddingProvider,
    chunks: Sequence[DocumentChunk],
    index: object,
    *,
    batch_size: int,
) -> None:
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = await provider.embed([chunk.embedding_text for chunk in batch])
        matrix = _validated_vectors(vectors, len(batch))
        if not isinstance(index, faiss.IndexFlatL2):
            raise IndexBuildError("Index builder created an unexpected FAISS index type.")
        index.add(matrix)  # pyright: ignore[reportUnknownMemberType] - FAISS exposes an incomplete overload.


def _write_faiss(path: Path, index: object) -> None:
    if not isinstance(index, faiss.IndexFlatL2):
        raise IndexBuildError("Index builder created an unexpected FAISS index type.")
    try:
        faiss.write_index(index, str(path))
    except (OSError, RuntimeError, TypeError) as error:
        raise IndexBuildError("Unable to stage the FAISS artifact.") from error


def _publish(staged: ArtifactPaths, destination: ArtifactPaths) -> None:
    """Replace all artifacts with rollback if a publication operation fails."""

    backup_directory = Path(tempfile.mkdtemp(prefix=".index-backup-", dir=destination.directory))
    destinations = (
        destination.faiss,
        destination.documents,
        destination.manifest,
        destination.directory / SQLITE_FILENAME,
    )
    staged_files = (staged.faiss, staged.documents, staged.manifest, staged.directory / SQLITE_FILENAME)
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for destination_path in destinations:
            if destination_path.exists():
                backup_path = backup_directory / destination_path.name
                os.replace(destination_path, backup_path)
                backups.append((destination_path, backup_path))
        for staged_path, destination_path in zip(staged_files, destinations, strict=True):
            if staged_path.exists():
                os.replace(staged_path, destination_path)
                published.append(destination_path)
    except (OSError, RuntimeError) as error:
        for destination_path in published:
            destination_path.unlink(missing_ok=True)
        for destination_path, backup_path in reversed(backups):
            if backup_path.exists():
                os.replace(backup_path, destination_path)
        raise IndexBuildError("Unable to publish the documentation index.") from error
    finally:
        shutil.rmtree(backup_directory, ignore_errors=True)


async def build_index(
    source_directory: Path,
    output_directory: Path,
    provider: EmbeddingProvider,
    provenance: EmbeddingProvenance,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
    overlap_characters: int = DEFAULT_OVERLAP_CHARACTERS,
) -> BuildResult:
    """Build a verified index and publish it as one artifact generation."""

    if type(provenance) is not EmbeddingProvenance:
        raise IndexBuildError("Index builder requires explicit immutable embedding provenance.")
    build_provenance = provenance

    source = source_directory.resolve()
    output = output_directory.resolve()
    items = load_source_items(source)
    chunks = chunk_items(
        items,
        max_characters=max_characters,
        overlap_characters=overlap_characters,
    )
    search_json_path = source / "search.json"
    search_json_sha256 = hash_file(search_json_path)
    html_tree_sha256 = hash_html_tree(source)

    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise IndexBuildError("Unable to create the index output directory.") from error
    if not output.is_dir():
        raise IndexBuildError("Index output must be a directory.")

    staging_directory = Path(tempfile.mkdtemp(prefix=".index-stage-", dir=output))
    staged = ArtifactPaths(staging_directory)
    try:
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        await _embed_chunks(provider, chunks, index, batch_size=batch_size)
        _write_faiss(staged.faiss, index)
        store = _document_store(chunks)
        write_document_store(staged.documents, store)
        write_lexical_index(staging_directory / SQLITE_FILENAME, staged.documents, store)
        load_lexical_artifacts(staging_directory)
        manifest = IndexManifest(
            schema_version=2,
            index_type="IndexFlatL2",
            distance_metric="l2",
            embedding_provider=build_provenance.provider.value,
            embedding_model=build_provenance.model,
            embedding_dimension=EMBEDDING_DIMENSION,
            document_count=len(chunks),
            search_json_sha256=search_json_sha256,
            html_tree_sha256=html_tree_sha256,
            faiss_sha256=hash_file(staged.faiss),
            documents_sha256=hash_file(staged.documents),
            chunking=ChunkingManifest(
                max_characters=max_characters,
                overlap_characters=overlap_characters,
            ),
        )
        write_index_manifest(staged.manifest, manifest)
        verified = load_verified_artifacts(staging_directory)
        if verified.manifest != manifest or verified.documents != store:
            raise IndexBuildError("Staged index artifacts changed during validation.")
        _publish(staged, ArtifactPaths(output))
    except TestAutomationSDKError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise IndexBuildError("Documentation index build failed before publication.") from error
    finally:
        shutil.rmtree(staging_directory, ignore_errors=True)

    return BuildResult(
        manifest=manifest,
        source_sections=len(items),
        chunk_count=len(chunks),
        output_directory=output,
    )


def build_lexical_index(source: Path, output: Path, *, from_documents: bool = False) -> int:
    """Publish a lexical generation from source exports or existing JSON chunks."""
    if from_documents:
        store = load_document_store(source)
    else:
        store = _document_store(chunk_items(load_source_items(source)))
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".index-stage-", dir=output))
    try:
        paths = ArtifactPaths(staging)
        write_document_store(paths.documents, store)
        write_lexical_index(staging / SQLITE_FILENAME, paths.documents, store)
        load_lexical_artifacts(staging)
        _publish(paths, ArtifactPaths(output))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return len(store.documents)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Test Automation SDK documentation index.")
    parser.add_argument("--source", type=Path, default=Path("data"), help="Raw documentation source directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
        help="Directory for the generated SQLite, documents, and optional vector artifacts.",
    )
    parser.add_argument(
        "--from-documents", type=Path, help="Build SQLite from existing documents JSON without raw sources."
    )
    return parser


async def _build_from_arguments(source: Path, output: Path) -> BuildResult:
    config = RuntimeConfig.from_environment()
    provider = create_embedding_provider(config)
    provenance = EmbeddingProvenance(config.provider, config.model)
    try:
        return await build_index(
            source,
            output,
            provider,
            provenance,
        )
    finally:
        await provider.aclose()


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the index-builder console command."""

    args = _parser().parse_args(arguments)
    print(f"Building documentation index from {args.source}...", file=sys.stderr)
    try:
        config = RuntimeConfig(search_mode="lexical") if args.from_documents else RuntimeConfig.from_environment()
        if args.from_documents is not None or config.search_mode == "lexical" or not config.embeddings_enabled:
            count = build_lexical_index(
                args.from_documents or args.source, args.output, from_documents=args.from_documents is not None
            )
            print(f"chunks={count} search=lexical output={args.output}")
            return 0
        try:
            result = asyncio.run(_build_from_arguments(args.source, args.output))
        except EmbeddingError:
            print("Embedding build unavailable; publishing a lexical index.", file=sys.stderr)
            count = build_lexical_index(args.source, args.output)
            print(f"chunks={count} search=lexical output={args.output}")
            return 0
    except TestAutomationSDKError as error:
        print(f"error: {error.safe_message}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: index build failed ({type(error).__name__})", file=sys.stderr)
        return 1
    manifest = result.manifest
    print(
        f"model={manifest.embedding_model} source_sections={result.source_sections} "
        f"chunks={result.chunk_count} vectors={manifest.document_count} "
        f"dimensions={manifest.embedding_dimension} output={result.output_directory}",
    )
    return 0


__all__ = ["BuildResult", "build_index", "build_lexical_index", "main"]
