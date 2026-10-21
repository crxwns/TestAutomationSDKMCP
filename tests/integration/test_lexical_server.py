"""Exercise packaged documentation through stdio with no inference configuration."""

import asyncio
import os
import sys
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def test_stdio_retrieval_without_embedding_endpoint(tmp_path: Path) -> None:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TA_SDK_")}
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "test_automation_sdk_mcp"],
        env=environment,
        cwd=tmp_path,
    )

    async def exercise() -> None:
        async with (
            stdio_client(parameters) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool("retrieve_documentation", {"query": "TestEnvironmentAccess"})
            assert not result.is_error
            assert result.structured_content is not None
            snippets = result.structured_content["result"]
            assert snippets
            assert all(snippet["search_method"] == "lexical" for snippet in snippets)
            assert all(snippet["score_type"] == "bm25" for snippet in snippets)
            assert all(snippet["chunk_id"] and snippet["location"] for snippet in snippets)

    asyncio.run(exercise())
