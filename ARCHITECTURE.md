# Test Automation SDK MCP Architecture

This document describes the implementation of the Test Automation SDK MCP
server. The server is a read-only, retrieval-augmented documentation service:
it exposes one MCP tool, `retrieve_documentation`, backed by SQLite FTS5.
An explicitly configured Ollama or OpenAI-compatible endpoint can optionally
turn queries into vectors. Document vectors are generated ahead of time.

The index is built ahead of time. Server startup loads and verifies the
published artifacts; it does not crawl the documentation or rebuild the index.

## Adaptive retrieval

`AdaptiveRetriever` composes SQLite retrieval and the existing vector-only
`DocumentationRetriever`. The default `auto` mode uses vectors when explicitly
configured and available; otherwise it searches SQLite. Expected semantic failures
start a monotonic-clock cooldown (60 seconds by default). Cancellation and unexpected
programming exceptions propagate. `hybrid` runs both retrieval paths concurrently
and combines candidate ranks by chunk ID with RRF (`k=60`), without comparing BM25
and L2 values. `lexical` never constructs or calls an embedding provider.

SQLite indexes the same ordered `DocumentStore` chunks as FAISS. Its metadata stores
the documents SHA-256 and schema version. Admission checks SQLite integrity, exact
ordered chunk contents, and FTS integrity without reading vector metadata. Optional
FAISS loading remains strict but failure leaves verified lexical artifacts available.
Legacy releases without SQLite can create an in-memory FTS snapshot from hash-verified
JSON. The runtime never writes index artifacts. Each adaptive retriever holds one
in-memory SQLite snapshot, reuses it for queries, and closes it on lifespan exit.

The builder can publish SQLite and documents alone, or include FAISS and its manifest
when an embedding URL is configured. Both paths stage, validate, and publish with
rollback. Lexical publication removes stale vector artifacts. `--from-documents`
reuses existing JSON chunks without raw sources or model calls. See the README for
configuration and output schema migration details. The vector-specific diagrams
below describe the optional semantic path.

## System Context

The MCP client is typically hosted by an AI agent. The agent decides when the
SDK-specific tool is needed, while the MCP server performs only deterministic
documentation retrieval. The selected embedding service supplies vectors but
does not generate the answer returned to the user.

```mermaid
flowchart LR
	 developer[Developer or user]
	 agent[AI agent]
	 client[MCP client host]
	 server[Test Automation SDK MCP server\nstdio transport]
	 tool[retrieve_documentation\nread-only structured tool]
	 artifacts[(Verified packaged or\nconfigured artifacts)]
	 embeddings[Selected embedding service\nOllama /api/embed or OpenAI /v1/embeddings]
	 docs[HTML documentation export\nsearch.json]

	 developer -->|SDK question or test task| agent
	 agent -->|tool decision and query| client
	 client <-->|MCP JSON-RPC over stdin/stdout| server
	 server --> tool
	 tool -->|load once, read-only| artifacts
	 tool -->|query embedding| embeddings
	 embeddings -->|finite 768-value vector| tool
	 tool -->|nearest sourced snippets| client
	 client --> agent
	 agent -->|answer grounded in snippets| developer
	 docs -. maintainer-only input .-> artifacts
```

### Runtime responsibilities

| Boundary                       | Responsibility                                                                                                                                                            |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MCP client host                | Starts the console entry point and forwards MCP requests from the agent.                                                                                                  |
| `test_automation_sdk_mcp.main` | Creates the server, configures stderr logging, and runs MCP over stdio.                                                                                                   |
| `create_server`                | Loads verified artifacts, registers the tool, and owns the provider lifecycle.                                                                                            |
| `DocumentationRetriever`       | Validates a query, requests its embedding, searches FAISS, validates rows and distances, and maps rows to documents without owning provider lifetime.                     |
| `provider/`                    | Selects Ollama or OpenAI-compatible transport, exposes `embed()` and asynchronous `aclose()`, validates wire boundaries, and translates failures into application errors. |
| `compatibility.py`             | Runs the advisory candidate-versus-index check by default or an explicit live provider parity comparison, then closes providers it owns.                                  |
| FAISS `IndexFlatL2`            | Stores the document embeddings and returns nearest row IDs with L2 distances.                                                                                             |
| `DocumentStore`                | Stores the ordered metadata and content records that correspond to FAISS row IDs.                                                                                         |

## Package Structure

```mermaid
flowchart TB
	 subgraph build[Build-time pipeline]
		  build_index[build_index.py\nbuild_index]
		  chunking[chunking.py\nvalidate, normalize, chunk, hash]
		  provider_build[provider/\nselected embedding provider]
		  documents[documents.py\nDocumentStore and IndexManifest]
		  index_io[index.py\nstage, verify, publish]
		  build_index --> chunking
		  build_index --> provider_build
		  build_index --> documents
		  build_index --> index_io
	 end

	 subgraph runtime[Runtime pipeline]
		  entry[__init__.py\nmain]
		  config[config.py\nRuntimeConfig]
		  server[server.py\ncreate_server and MCP tool]
		  retriever[DocumentationRetriever]
		  provider_runtime[provider/\nasync HTTP providers]
		  index_runtime[index.py\nload_verified_artifacts]
		  errors[errors.py\npublic MCP error envelope]
		  entry --> server
		  config --> server
		  server --> retriever
		  server --> provider_runtime
		  server --> index_runtime
		  server --> errors
		  retriever --> provider_runtime
		  retriever --> index_runtime
	 end

	 build_index -. publishes .-> index_runtime
	 provider_build -. same embedding contract .-> provider_runtime
```

The build and runtime paths share the embedding dimension and provider contract,
but they have different jobs. The builder creates artifacts; the runtime only
loads, validates, and queries them. Provider/model values in the manifest are
build provenance and do not serve as runtime admission keys.

## Compatibility Evaluation

The compatibility console command has two distinct advisory paths. Index mode
is the default and needs only the configured candidate endpoint. It compares
candidate embeddings for deterministic document samples with the stored vectors
from the verified artifact generation, searches that same index with candidate
query probes, and checks representative documentation locations.

Parity mode is opt-in and needs two active providers. It embeds the same mixed
query and document probe corpus through a baseline and candidate provider, then
reports same-input and geometry similarity alongside retrieval overlap. It is a
diagnostic comparison between live providers, not a runtime admission check.

```mermaid
flowchart LR
	corpus[Deterministic probe corpus]
	artifacts[(Verified artifact generation)]
	candidate[Selected candidate provider]
	baseline[Optional live baseline provider]
	index_mode[Default index mode]
	parity_mode[Opt-in parity mode]
	index_report[Index compatibility report]
	parity_report[Live parity report]

	corpus --> index_mode
	corpus --> parity_mode
	artifacts --> index_mode
	artifacts --> parity_mode
	candidate --> index_mode
	candidate --> parity_mode
	baseline --> parity_mode
	index_mode --> index_report
	parity_mode --> parity_report
```

Index mode uses the verified generation itself as the document-space reference;
it does not publish a duplicate reference sidecar or require an Ollama service.
Parity mode retains the stronger same-input query comparison for cases where a
live provider-to-provider measurement is useful. Neither mode writes approval
state, changes runtime configuration, or changes the published artifacts.

## Building the Index

The source directory must contain an HTML tree and `search.json`. The builder
requires exact page coverage between those inputs, converts search records from
HTML fragments to normalized Markdown, and chunks each section using stable
character limits and overlap. Each chunk retains its source location, title,
breadcrumbs, tags, and chunk index.

```mermaid
flowchart TD
	 source[HTML documentation tree\nplus search.json]
	 load[load_source_items\nparse and validate search records]
	 coverage[validate_source_coverage\nHTML pages exactly match locations]
	 normalize[normalize_html_fragment\nHTML fragment to Markdown]
	 chunk[chunk_items\nsplit at textual boundaries\nwith overlap]
	 embedding_text[Build embedding text\nbreadcrumbs + title + content]
	 embed[Selected provider.embed\nbatched provider requests]
	 validate_vectors[Validate count, finite values,\nand 768 dimensions]
	faiss["IndexFlatL2(768)\nadd vectors in chunk order"]
	 store[DocumentStore\nrecords in the same order as vectors]
	 hashes[Hash source search.json\nand HTML tree]
	 manifest[IndexManifest\nmodel, metric, counts, chunking, hashes]
	 stage[Temporary staging directory\nwrite shared chunks and indexes]
	 verify[load_verified_artifacts\nvalidate schema, hashes, dimensions,\ncounts, metric, and row mapping]
	 publish[Atomic publish\nbackup old generation, replace all files,\nrollback on failure]
	 artifacts[(TA_Docu.faiss\nTA_Docu.documents.json\nTA_Docu.manifest.json)]

	 source --> load --> coverage --> normalize --> chunk --> embedding_text --> embed --> validate_vectors --> faiss
	 chunk --> store
	 source --> hashes
	 faiss --> stage
	 store --> stage
	 hashes --> manifest
	 validate_vectors --> manifest
	 manifest --> stage
	 stage --> verify --> publish --> artifacts
```

### Build steps

1. `load_source_items` parses `search.json` and checks that every indexed page
	exists in the HTML tree and every HTML page is represented.
2. `chunk_items` normalizes each section and produces deterministic chunk IDs
	from location, chunk index, and content. The default maximum is 1,000
	characters with 200 characters of overlap.
3. `_embed_chunks` sends embedding text to the selected provider in batches.
	The default batch size is 32. Every response must contain one finite vector
	of exactly 768 values per input.
4. FAISS receives vectors in the same order as the `DocumentStore` records.
	This order is the row-ID contract used during retrieval.
5. The builder writes SQLite, FAISS, document JSON, and the vector manifest in a
	temporary directory, reloads them through the normal verification path, and
	publishes the complete generation together. A failed publication restores
	the previous generation.

## Published Artifact Contract

A lexical generation consists of `TA_Docu.documents.json` and `TA_Docu.sqlite3`.
A vector generation adds the two vector artifacts. The vector contract is:

```mermaid
flowchart LR
	 manifest[TA_Docu.manifest.json\nsource and artifact hashes\nmodel and vector contract]
	 faiss[TA_Docu.faiss\nFAISS IndexFlatL2\nrow ID -> vector]
	 documents[TA_Docu.documents.json\nordered DocumentRecord list\nrow ID -> content and metadata]
	 manifest -->|document_count and hashes| faiss
	 manifest -->|document_count and hashes| documents
	 faiss <-->|same row order and count| documents
```

At runtime, `load_verified_artifacts` rejects missing files, invalid JSON or
schemas, hash mismatches, mixed generations, wrong FAISS dimensions, wrong
metrics, and count mismatches. The configured provider and model may differ
from the manifest because those manifest fields describe build provenance.
Embedding-space compatibility is an operator decision. The default
compatibility check compares the candidate with this verified generation;
optional parity mode compares two live providers. Neither path changes startup
or runtime admission.

## Server Startup and Lifespan

```mermaid
sequenceDiagram
	 participant Host as MCP client host
	 participant Main as main()
	 participant Config as RuntimeConfig
	 participant Loader as artifact loader
	 participant Server as MCPServer
	 participant Provider as Selected embedding provider

	 Host->>Main: start console script
	 Main->>Config: read TA_SDK_* environment
	 Config-->>Main: validated endpoint, model, timeouts, result count, artifact directory
	 Main->>Loader: load packaged or configured artifacts
	 Loader-->>Main: verified LoadedArtifacts
	 Main->>Server: create_server and register tool
	 Main->>Server: run(transport="stdio")
	 Server->>Provider: create provider in lifespan
	 Server-->>Host: MCP initialize and tool listing
	 Note over Server,Provider: Provider is shared by retrieval calls during this server lifespan.
	 Host->>Server: shutdown
	 Server->>Provider: aclose()
```

The default artifact directory is the package's `db/` resource directory. A
filesystem directory can be selected with `TA_SDK_DB_DIR`. The runtime never
uses the current working directory to discover packaged artifacts.

Provider lifecycle is managed at composition roots. The server lifespan, index
builder entry point, and compatibility checker close the providers they create
or receive in their cleanup paths. `DocumentationRetriever` only consumes a
provider and never closes it. Providers without owned resources still satisfy
the shared contract with a no-op asynchronous `aclose()` implementation.

## Retrieval and Agent Tool Invocation

The agent should call the tool for authoritative Test Automation SDK questions
and SDK-based Python test development. Generic Python questions and unrelated
dSPACE products are outside this server's intended scope.

```mermaid
sequenceDiagram
	 autonumber
	 participant User as Developer or user
	 participant Agent as AI agent
	 participant Client as MCP client
	 participant MCP as MCP server
	 participant Tool as retrieve_documentation
	 participant Provider as Selected embedding provider
	 participant Embeddings as Ollama or OpenAI-compatible endpoint
	 participant FAISS as IndexFlatL2
	 participant Store as DocumentStore

	 User->>Agent: Ask SDK-specific question or describe test task
	 Agent->>Agent: Decide documentation retrieval is needed
	 Agent->>Client: Call tool with one self-contained query
	 Client->>MCP: JSON-RPC tools/call over stdio
	 MCP->>Tool: validate structured query
	 Tool->>Provider: embed([query])
	 Provider->>Embeddings: POST provider embeddings endpoint
	 Embeddings-->>Provider: one finite 768-value embedding
	 Provider-->>Tool: validated float32 query matrix
	 Tool->>FAISS: search(query_vector, min(result_count, ntotal, document_count))
	 FAISS-->>Tool: distances and row IDs
	 Tool->>Store: map each row ID to ordered DocumentRecord
	 Store-->>Tool: content, title, location, breadcrumbs
	 Tool-->>MCP: nearest-first DocumentationSnippet list
	 MCP-->>Client: structured MCP result with typed scores
	 Client-->>Agent: sourced snippets
	 Agent-->>User: answer grounded in returned locations and content
```

### Retrieval details

1. The query must be a string, non-empty after trimming, and no longer than
	4,000 characters.
2. The provider validates the returned shape as `(1, 768)` and rejects
	non-finite values.
3. The result count is bounded by the configured value, the number of FAISS
	vectors, and the number of document records. It defaults to 5 and is capped
	at 50 by configuration.
4. FAISS searches with the L2 metric. Results are sorted nearest-first by
	distance, and every row ID is checked before it is used as a document index.
5. The tool returns `chunk_id`, `content`, `location`, `title`, `breadcrumbs`,
	`search_method`, `score`, and `score_type` for each snippet. Vector scores are
	uncalibrated L2; lexical scores are BM25; fused scores are RRF. It does not call a
	chat model and does not synthesize the final answer.

## Error and Trust Boundaries

The server validates data at each external boundary: environment variables,
source files, provider requests and responses, FAISS results, artifact files, and
MCP tool output. Expected application failures are translated to a safe native
MCP error with a code, classification, and retryability flag.

| Boundary            | Examples                                                         | Runtime behavior                                                                 |
| ------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Query input         | Empty, non-string, or oversized query                            | `invalid_query`; permanent and not retryable.                                    |
| Provider connection | Timeout or unavailable endpoint                                  | SQLite fallback and a semantic cooldown.        |
| Provider response   | Wrong response model, indexes, vector count, dimension, or value | SQLite fallback and a semantic cooldown; correct the provider contract. |
| FAISS/artifacts     | Invalid index, row ID, distance, hash, count, or schema          | Retrieval or artifact failure; the server does not serve unverified data.        |
| MCP output          | Invalid snippet fields or distance                               | `tool_output_invalid`; the response is rejected rather than returned partially.  |

Secrets such as `TA_SDK_OLLAMA_API_KEY` and `TA_SDK_OPENAI_API_KEY` are used
only as HTTP bearer tokens and are not included in logs or public error messages.
Protocol messages stay
on stdout; human-readable logging and startup errors go to stderr so stdio MCP
traffic remains parseable.

## Configuration Surface

| Variable                    | Default                  | Used by                                                     |
| --------------------------- | ------------------------ | ----------------------------------------------------------- |
| `TA_SDK_SEARCH_MODE` | `auto` | `auto`, `hybrid`, or `lexical`. |
| `TA_SDK_EMBEDDING_COOLDOWN` | `60` seconds | Semantic retry cooldown after an expected failure. |
| `TA_SDK_EMBEDDING_PROVIDER` | `ollama`                 | Selects `ollama` or `openai`.                               |
| `TA_SDK_OLLAMA_URL`         | unset | Ollama base URL; `/api/embed` is appended.                  |
| `TA_SDK_OLLAMA_MODEL`       | `nomic-embed-text:v1.5`  | Required request model; manifest value is build provenance. |
| `TA_SDK_OLLAMA_API_KEY`     | unset                    | Optional Ollama bearer token.                               |
| `TA_SDK_OPENAI_URL`         | unset                    | Complete `/v1/embeddings` endpoint.                         |
| `TA_SDK_OPENAI_MODEL`       | unset                    | Optional request model; not an artifact admission key.      |
| `TA_SDK_OPENAI_API_KEY`     | unset                    | Optional OpenAI bearer token.                               |
| `TA_SDK_RESULT_COUNT`       | `5`                      | Maximum nearest snippets, from 1 through 50.                |
| `TA_SDK_DB_DIR`             | packaged `db/`           | Alternate fully validated artifact directory.               |
| `TA_SDK_CONNECT_TIMEOUT`    | `5` seconds              | HTTP connection timeout.                                    |
| `TA_SDK_REQUEST_TIMEOUT`    | `30` seconds             | HTTP request timeout.                                       |

## Source Map

- `src/test_automation_sdk_mcp/__init__.py`: console entry point and stdio
  process lifecycle.
- `src/test_automation_sdk_mcp/server.py`: MCP server construction, tool
  schema, retrieval orchestration, and structured output.
- `src/test_automation_sdk_mcp/build_index.py`: source-to-artifact build and
  atomic publication.
- `src/test_automation_sdk_mcp/chunking.py`: source validation, HTML
  normalization, chunking, and source hashes.
- `src/test_automation_sdk_mcp/provider/`: shared embedding contract, Ollama
	and OpenAI-compatible transports, and the provider factory.
- `src/test_automation_sdk_mcp/compatibility.py`: optional advisory embedding
	space comparison and safe report generation.
- `src/test_automation_sdk_mcp/ollama.py`: compatibility imports for the
	legacy Ollama provider module.
- `src/test_automation_sdk_mcp/index.py`: artifact I/O and cross-file
  generation validation.
- `src/test_automation_sdk_mcp/documents.py`: strict document and manifest
  models.
- `src/test_automation_sdk_mcp/config.py`: environment configuration and
  validation.
- `src/test_automation_sdk_mcp/errors.py`: safe application-to-MCP error
  translation.
