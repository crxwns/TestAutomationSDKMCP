import asyncio
import json
import os
import tempfile
from collections.abc import Awaitable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import TextIO, cast

import numpy as np
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from numpy.typing import NDArray

from test_automation_sdk_mcp.build_index import build_index
from test_automation_sdk_mcp.config import EmbeddingProviderKind
from test_automation_sdk_mcp.documents import load_document_store
from test_automation_sdk_mcp.provider import EmbeddingProvenance


class StaticEmbeddingProvider:
    async def embed(self, inputs: Sequence[str]) -> NDArray[np.float32]:
        return np.zeros((len(inputs), 768), dtype=np.float32)

    async def aclose(self) -> None:
        return None


class EmbeddingHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        response = json.dumps({"model": "fake-model", "embeddings": [[0.0] * 768]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        return None


def run[T](coroutine: Awaitable[T]) -> T:
    return asyncio.run(coroutine)


def test_installed_stdio_entry_point_supports_client_session_without_stdout_noise(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source = tmp_path / "source"
    source.mkdir()
    (source / "index.html").write_text("raw html", encoding="utf-8")
    (source / "search.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "location": "index.html",
                        "level": 1,
                        "title": "Intro",
                        "text": "<p>Body</p>",
                        "path": [],
                        "tags": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "db"
    run(
        build_index(
            source,
            output,
            StaticEmbeddingProvider(),
            EmbeddingProvenance(EmbeddingProviderKind.OLLAMA, "fake-model"),
        )
    )

    environment = {key: value for key, value in os.environ.items() if not key.startswith("TA_SDK_")}
    environment.update(
        {
            "TA_SDK_DB_DIR": str(output),
            "TA_SDK_OLLAMA_MODEL": "fake-model",
        }
    )
    embedding_server = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingHandler)
    embedding_thread = Thread(target=embedding_server.serve_forever, daemon=True)
    embedding_thread.start()
    environment["TA_SDK_OLLAMA_URL"] = f"http://127.0.0.1:{embedding_server.server_port}"
    parameters = StdioServerParameters(
        command="uv",
        args=["run", "test-automation-sdk-mcp"],
        env=environment,
        cwd=repository_root,
    )

    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_handle:
            stderr = cast(TextIO, stderr_handle)

            async def exercise() -> tuple[list[str], object, bool]:
                async with (
                    stdio_client(parameters, errlog=stderr) as (read_stream, write_stream),
                    ClientSession(read_stream, write_stream) as session,
                ):
                    await session.initialize()
                    tools = await session.list_tools()
                    result = await session.call_tool("retrieve_documentation", {"query": "intro"})
                    return [tool.name for tool in tools.tools], result.structured_content, result.is_error

            tool_names, structured_content, is_error = run(exercise())
            stderr.seek(0)
            stderr_output = stderr.read()
    finally:
        embedding_server.shutdown()
        embedding_server.server_close()
        embedding_thread.join(timeout=5)

    assert tool_names == ["retrieve_documentation"]
    assert structured_content == {
        "result": [
            {
                "content": "Body",
                "location": "index.html",
                "title": "Intro",
                "breadcrumbs": [],
                "chunk_id": load_document_store(output / "TA_Docu.documents.json").documents[0].id,
                "search_method": "vector",
                "score": 0.0,
                "score_type": "l2",
            }
        ]
    }
    assert is_error is False
    assert "error:" not in stderr_output.lower()
