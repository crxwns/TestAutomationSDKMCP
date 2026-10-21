"""SQLite FTS5 retrieval, independently validated against the shared chunk store."""

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from .chunking import hash_file
from .documents import DocumentStore, load_document_store, load_index_manifest
from .errors import ArtifactError
from .index import ArtifactPaths, LoadedArtifacts


@dataclass(frozen=True, slots=True)
class LexicalArtifacts:
    documents: DocumentStore
    paths: ArtifactPaths
    index: None = None


SQLITE_FILENAME = "TA_Docu.sqlite3"


def _rows(store: DocumentStore) -> list[tuple[str, str, str, str, str]]:
    return [(doc.id, doc.title, " ".join(doc.breadcrumbs), " ".join(doc.tags), doc.content) for doc in store.documents]


def _populate(connection: sqlite3.Connection, store: DocumentStore) -> None:
    connection.execute(
        "CREATE VIRTUAL TABLE chunks USING fts5(id UNINDEXED, title, breadcrumbs, tags, content, "
        "tokenize='unicode61 remove_diacritics 2')"
    )
    connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)", _rows(store))


def write_lexical_index(path: Path, documents_path: Path, store: DocumentStore) -> None:
    """Build from the same ordered chunks used for FAISS, without embedding calls."""
    try:
        with closing(sqlite3.connect(path)) as connection, connection:
            _populate(connection, store)
            connection.execute("CREATE TABLE metadata (schema_version INTEGER, documents_sha256 TEXT)")
            connection.execute("INSERT INTO metadata VALUES (1, ?)", (hash_file(documents_path),))
    except sqlite3.Error as error:
        raise ArtifactError("Unable to build SQLite documentation index (FTS5 is required).") from error


def load_lexical_artifacts(directory: Path) -> LexicalArtifacts:
    """Validate SQLite and documents without opening the optional vector artifacts."""
    paths = ArtifactPaths(directory)
    store = load_document_store(paths.documents)
    sqlite_path = directory / SQLITE_FILENAME
    try:
        with closing(sqlite3.connect(f"{sqlite_path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ArtifactError("SQLite documentation index failed its integrity check.")
            if connection.execute("SELECT * FROM metadata").fetchall() != [(1, hash_file(paths.documents))]:
                raise ArtifactError("SQLite index and document generation do not match.")
            if connection.execute(
                "SELECT id, title, breadcrumbs, tags, content FROM chunks ORDER BY rowid"
            ).fetchall() != _rows(store):
                raise ArtifactError("SQLite chunks do not match the document store.")
            # Verify the inverted index as well as the visible FTS content, on an in-memory copy.
            with closing(sqlite3.connect(":memory:")) as check:
                connection.backup(check)
                check.execute("INSERT INTO chunks(chunks) VALUES ('integrity-check')")
    except sqlite3.Error as error:
        raise ArtifactError("SQLite documentation index is missing or invalid.") from error
    return LexicalArtifacts(store, paths)


def load_runtime_artifacts(directory: Path, *, vectors: bool) -> LoadedArtifacts | LexicalArtifacts:
    """Admit lexical artifacts first; vector failure cannot prevent lexical startup.

    Older releases have no SQLite artifact. Their hash-verified JSON chunks can
    still back an in-memory FTS index without modifying the installed package.
    """
    from .index import load_verified_artifacts

    if (directory / SQLITE_FILENAME).exists():
        lexical = load_lexical_artifacts(directory)
    else:
        paths = ArtifactPaths(directory)
        manifest = load_index_manifest(paths.manifest)
        store = load_document_store(paths.documents)
        if manifest.documents_sha256 != hash_file(paths.documents) or manifest.document_count != len(store.documents):
            raise ArtifactError("Document store does not match the index manifest.")
        lexical = LexicalArtifacts(store, paths)
    if vectors:
        try:
            return load_verified_artifacts(directory)
        except ArtifactError:
            pass
    return lexical


class LexicalSearch:
    """One immutable in-memory FTS snapshot, reused for the retriever's lifetime."""

    def __init__(self, store: DocumentStore) -> None:
        self._connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = Lock()
        try:
            _populate(self._connection, store)
        except sqlite3.Error as error:
            self._connection.close()
            raise ArtifactError("Unable to initialize SQLite search (FTS5 is required).") from error

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def search(self, query: str, count: int) -> list[tuple[str, float]]:
        """Treat user text as literal tokens, never as an FTS expression."""
        tokens = list(dict.fromkeys(re.findall(r"\w+", query, re.UNICODE)))
        if not tokens:
            return []
        expression = " OR ".join('"' + token + '"' for token in tokens)
        try:
            with self._lock:
                rows = self._connection.execute(
                    "SELECT id, bm25(chunks, 0, 5, 2, 2, 1) AS score FROM chunks "
                    "WHERE chunks MATCH ? ORDER BY score, rowid LIMIT ?",
                    (expression, count),
                ).fetchall()
        except sqlite3.Error as error:
            from .errors import ApplicationErrorCode, RetrievalError

            raise RetrievalError(
                "SQLite documentation search failed.", code=ApplicationErrorCode.RETRIEVAL_FAILED
            ) from error
        return [(str(row[0]), float(row[1])) for row in rows]
