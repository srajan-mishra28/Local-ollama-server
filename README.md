# Local Ollama Server

A FastAPI gateway in front of a locally running [Ollama](https://ollama.com)
model. It adds the things the raw Ollama API does not give you: persistent chat
sessions, file transfer from other devices on the LAN, and the ability to pin
source files from a project as context for a conversation.

Runs on one machine, reachable from any device on the same network.

## Why

Ollama exposes a model over HTTP but keeps no history, has no UI worth using from
a phone, and has no notion of "this conversation is about that codebase". This
server sits in between and handles all three, while keeping Ollama itself bound to
localhost.

## Features

- **Persistent sessions.** Chat history is stored in SQLite, so conversations
  survive restarts. Sessions are listed in a sidebar and named after their first
  message.
- **Project context.** Register a folder on disk, then tick the files that should
  be injected as a system message for a session. Includes a path filter and a
  content search for locating the right files.
- **Project RAG.** Index a project into a local vector store (Ollama embeddings,
  SQLite + numpy, no extra server). At chat time the most relevant chunks for
  your question are retrieved and injected alongside any pinned files. Besides
  text and code it indexes **PDFs** (text layer, with page markers) and
  **images** (described by an Ollama vision model).
- **File transfer.** Upload from a phone or tablet straight into `uploads/` on the
  host, then download or delete from the same panel.
- **Streaming responses.** Tokens are streamed to the browser as they are
  generated, and the full reply is persisted once the stream completes.
- **Modern, mobile-friendly UI.** Served from `/`, no build step, no CDN (works
  offline on a LAN). Light and dark themes (follows the system, with a manual
  toggle that is remembered), markdown rendering with copyable code blocks,
  auto-growing input (Enter sends, Shift+Enter adds a line), toasts for errors
  such as Ollama being down, and an index panel per project: index button with a
  result summary, indexed files with chunk counts, per-file re-sync/remove, and
  uploads of PDFs, images and text.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- A pulled chat model (default is `qwen2.5-coder:7b`)
- For project RAG: an embedding model (`nomic-embed-text`)
- Optional, for images in project RAG: a vision model (default `qwen2.5vl:7b`,
  needs a recent Ollama)

## Setup

```bash
# 1. Pull a model
ollama pull qwen2.5-coder:7b

#    and an embedding model for project RAG
ollama pull nomic-embed-text

#    and a vision model if you want images indexed (set VISION_MODEL = "" to skip)
ollama pull qwen2.5vl:7b

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

### Access from other devices

Find the host's LAN address (`ipconfig` on Windows, `ip addr` on Linux) and open
`http://<lan-ip>:8000` from any device on the same network.

On Windows, allow the port through the firewall for private networks only:

```powershell
New-NetFirewallRule -DisplayName "Local Ollama Server" -Direction Inbound `
  -LocalPort 8000 -Protocol TCP -Action Allow -Profile Private
```

Leave Ollama itself bound to `localhost`. Only this server needs to be reachable,
which keeps a single place to add authentication later.

## Project layout

```
app.py               entry point: builds the FastAPI app, mounts routers and static files
rag.py               RAG vector store (PDF/image extraction, chunking, embeddings, SQLite + numpy)
backend/
  config.py          paths and tunable constants
  database.py        SQLite connection helper and schema
  rag_store.py       the shared RagStore instance
  routes/
    sessions.py      chat sessions, context, project binding
    projects.py      project folders, file listing, content search
    rag_index.py     per-project index, upload and retrieval endpoints
    files.py         file transfer to/from uploads/
    chat.py          prompt assembly, streamed Ollama reply, /api/health
static/
  index.html         the web UI markup
  style.css          UI styles
  app.js             UI logic
tests/               pytest suite
```

The UI is plain static files served from `/static`; there is no build step.
Run the server with `python -m uvicorn app:app` from the repository root.

## How it works

```
Browser (static/*)  ──HTTP──▶  FastAPI (app.py + backend/routes/*)  ──HTTP──▶  Ollama
                                   │                                  /api/chat  (answers)
                                   ├─ server.db   sessions, messages, projects
                                   ├─ uploads/    files transferred from devices
                                   └─ rag.db + project_files/<id>/   RAG index + file copies
                                                                      /api/embed (vectors)
```

**Chat request (`POST /api/chat`)**

1. The user message is saved to `server.db`; the session is titled after its
   first message.
2. The prompt is assembled, in order:
   1. *Pinned files*: files ticked in the UI, read from the project folder
      (capped by `MAX_CONTEXT_BYTES`), as a system message.
   2. *RAG excerpts*: the question is embedded, compared with the project's
      stored chunks, and the top `RAG_TOP_K` are added as a second system message.
      Skipped if the session has no project, the project is not indexed, or
      Ollama or the index fails.
   3. The session's message history, then the new message.
3. The prompt is streamed to Ollama; tokens are relayed to the browser as they
   arrive, and the full reply is saved when the stream ends.

**Indexing (`POST /api/projects/{id}/index`)**

1. The project folder is scanned (`CODE_EXT`, PDF and image files, `SKIP_DIRS` excluded).
2. Each file is hashed together with `EMBED_MODEL` (images also with
   `VISION_MODEL`, so changing it re-describes them); unchanged files are skipped.
3. Changed files are turned into text, split into ~1500-character chunks with 200
   overlap, and embedded through Ollama's `/api/embed`:
   text files as-is; PDFs per page with a `[page N]` marker at the start of every
   chunk; images via the vision model (`/api/chat` with a base64 image, asked to
   transcribe visible text and describe the content), each chunk starting with
   `[image: <path>]`.
4. Chunks and vectors go to `rag.db` (normalised float32 blobs). Only after
   embedding succeeds are the file's old chunks replaced and its copy written to
   `project_files/<id>/`.
5. Files that vanished from the folder are pruned from the index.

Retrieval is a cosine search (a numpy dot product) over that project's chunks.
In the UI use Projects, then Index on a project, then "Index project" after
registering it and whenever files change (or
`curl -X POST http://localhost:8000/api/projects/<id>/index`).

### Supported file types

| Type | Extensions | How it is indexed | Size cap |
| --- | --- | --- | --- |
| Text / code | `CODE_EXT` (`.py`, `.java`, `.md`, `.json`, ...) | UTF-8 text | 1 MB |
| PDF | `.pdf` | Text layer via `pypdf`, page by page | 20 MB, first 300 pages |
| Image | `.png` `.jpg` `.jpeg` `.webp` `.gif` `.bmp` | Described by `VISION_MODEL` | 10 MB |

Size is checked before a file is read into memory (folder scan) or buffered
(upload). Outcomes that are skipped rather than errors:

| Result | Meaning |
| --- | --- |
| `skipped` | Empty, binary, or over the size cap |
| `skipped_pdf_no_text` | PDF without a text layer (a scan); there is no OCR for PDFs |
| `skipped_pdf_unreadable` | PDF is encrypted (password needed) or corrupt |
| `skipped_image_failed` | The vision model is missing or failed on this image; not remembered, so the next index retries it |
| `skipped_image_disabled` | `VISION_MODEL` is empty |
| `captioned` | An image was described and indexed |

The index response keeps `indexed`, `skipped`, etc. as totals and adds the specific
keys above when they occur. If Ollama itself is unreachable the run stops with 502
(vision-model errors alone do not stop it). Indexing a large image folder means one
vision call per image, which can be slow on CPU; set `VISION_MODEL = ""` to turn
images off (they are then not scanned at all). Turning it off is non-destructive:
images captioned earlier stay in the index and remain searchable; they are just no
longer refreshed or pruned.

`webp` and `bmp` support depends on your Ollama version; PNG/JPEG/GIF are the safe choice.
PDFs are parsed with `pypdf` in a worker thread with no timeout, so a hostile PDF can
tie up a worker (size, page and text caps apply); fine for a LAN tool, but only
index or upload PDFs you trust.

Pinned context stays text-only: PDFs and images are not listed for pinning and are
never injected raw; their extracted text or description reaches the chat through
retrieval once indexed. Excerpts shown to the model carry their file path, and
`[page N]` / `[image: ...]` markers where applicable.

## Configuration

Edit the constants in `backend/config.py`:

| Constant | Default | Purpose |
| --- | --- | --- |
| `OLLAMA` | `http://localhost:11434` | Where Ollama is listening |
| `MODEL` | `qwen2.5-coder:7b` | Model used for all sessions |
| `MAX_CONTEXT_BYTES` | `120000` | Cap on injected project context |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model used for RAG embeddings |
| `VISION_MODEL` | `qwen2.5vl:7b` | Ollama vision model for images; `""` disables image indexing |
| `VISION_TIMEOUT` | `300` | Seconds to wait for one image description |
| `PDF_MAX_BYTES` | `20000000` | Largest PDF indexed or uploaded |
| `IMAGE_MAX_BYTES` | `10000000` | Largest image indexed or uploaded |
| `PDF_MAX_PAGES` | `300` | Pages read per PDF |
| `RAG_TOP_K` | `5` | Chunks injected per question |
| `RAG_CHUNK_CHARS` / `RAG_OVERLAP_CHARS` | `1500` / `200` | Chunk size and overlap |
| `CODE_EXT` | see source | Extensions eligible as context |
| `SKIP_DIRS` | see source | Directories excluded when scanning a project |

Raise `MAX_CONTEXT_BYTES` only alongside the context length configured in Ollama,
otherwise the prompt is silently truncated by the model.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness and active model |
| `GET` | `/api/sessions` | List sessions |
| `POST` | `/api/sessions` | Create a session |
| `GET` | `/api/sessions/{id}` | Session with full message history |
| `DELETE` | `/api/sessions/{id}` | Delete a session and its messages |
| `PUT` | `/api/sessions/{id}/context` | Set the context file list |
| `PUT` | `/api/sessions/{id}/project` | Bind a project to a session |
| `GET` | `/api/projects` | List registered projects |
| `POST` | `/api/projects` | Register a folder |
| `DELETE` | `/api/projects/{id}` | Unregister a project |
| `GET` | `/api/projects/{id}/files` | List candidate context files |
| `GET` | `/api/projects/{id}/grep` | Find files containing a term |
| `POST` | `/api/projects/{id}/index` | (Re)index the project folder; incremental, drops vanished files |
| `GET` | `/api/projects/{id}/index` | Indexed files and chunk counts |
| `POST` | `/api/projects/{id}/index/file?path=` | Re-sync one file (re-index, or remove if deleted) |
| `DELETE` | `/api/projects/{id}/index/file?path=` | Remove a file's chunks and local copy |
| `POST` | `/api/projects/{id}/upload` | Upload a file into the project and index it |
| `GET` | `/api/projects/{id}/rag?q=&k=` | Query the index directly |
| `POST` | `/api/chat` | Send a message, stream the reply |
| `POST` | `/api/upload` | Upload a file |
| `GET` | `/api/files` | List uploaded files |
| `GET` | `/api/files/{name}` | Download a file |
| `DELETE` | `/api/files/{name}` | Delete a file |

Example:

```bash
curl -N http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": 1, "content": "explain CDI scopes"}'
```

## Data

All are created on first run in the repository root and are gitignored:

- `server.db` holds projects, sessions and messages
- `uploads/` holds transferred files
- `rag.db` holds chunks and embeddings; `project_files/<id>/` holds the local
  copy of every indexed file (all gitignored)

Indexing is explicit: call `POST /api/projects/{id}/index` after registering a
project and again after files change (unchanged files are skipped by hash).
Uploads through `/api/projects/{id}/upload` are indexed immediately and are not
removed by a re-index. If Ollama or the embedding model is unavailable, chat
still works, just without retrieved excerpts.

Notes on behaviour:

- A path uploaded through `/upload` wins over the same path in the project
  folder: a later index will not overwrite or remove it (reported as `shadowed`).
- Uploads over the type's cap (1 MB text, 20 MB PDF, 10 MB image) are rejected with
  413; other binary files are accepted but not indexed or stored.
- If you change `EMBED_MODEL` (or its vector size), old chunks are ignored and
  `/rag` answers 409 until you run the index again.
- If Ollama is unreachable, `POST .../index` stops at the first failure and
  returns 502 instead of failing every file.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

Tests use a fake embedder and a fake captioner, and build their PDFs by hand, so
Ollama is not needed.

## Security

This server has **no authentication**. It is built for a trusted home network.
Before exposing it anywhere else, add an API key check or put a reverse proxy
with TLS in front of it. Uploaded filenames are flattened to their basename and
project context reads are confined to the registered project root, but neither
substitutes for access control. Because there is no auth, any page open
in a browser on the same network can also call the write endpoints (CSRF);
retrieved file excerpts are fed to the model as untrusted reference text.

## Licence

MIT
