# Local Ollama Server

<img width="1917" height="955" alt="image" src="https://github.com/user-attachments/assets/eb25aa07-cc10-47ba-86b9-f54a821b9fff" />

A FastAPI gateway in front of a locally running [Ollama](https://ollama.com)
model. It adds the things the raw Ollama API does not give you: persistent chat
sessions, projects whose chats share uploaded files through a local RAG index,
and file transfer from other devices on the LAN.

Runs on one machine, reachable from any device on the same network.

## Why

Ollama exposes a model over HTTP but keeps no history, has no UI worth using from
a phone, and has no notion of "these conversations are about those documents". This
server sits in between and handles all three, while keeping Ollama itself bound to
localhost.

## Concepts: Chats and Projects

- **Chats** are plain conversations. They have a history and nothing else: no
  files, no retrieval. They are listed under **Chats** in the sidebar.
- **Projects** are named workspaces (not folders on disk). A project has its own
  **chats** and its own **files**. Create one by name, open it, upload files, and
  start chats inside it.

**How sharing works.** Every file uploaded to a project goes straight through the
RAG pipeline: a copy is kept in `project_files/<id>/`, the text is split into
chunks, and the chunks are embedded (Ollama) into `rag.db`. All chats of that
project retrieve from the same index. For each question the most relevant chunks
are added to the prompt, so a project can know far more than any single chat's
context window can hold, and a new chat starts with all of it. Uploading is the
only way to add files; there is no separate index step. Files uploaded later are
available to existing chats immediately.

**Two limits keep prompts inside the model's window** (both in
`backend/config.py`): retrieved excerpts are capped at `RAG_MAX_EXCERPT_CHARS`
in total, and a chat's history is cut to the newest messages that fit in
`MAX_HISTORY_CHARS` (the latest message is always sent). Older turns stay stored
and visible in the UI; they just are not sent to the model. In a project, what
older turns contained should live in the files.

## Features

- **Persistent chats.** History is stored in SQLite, so conversations survive
  restarts. Chats are named after their first message.
- **Projects with shared knowledge.** Multi-file upload (button or drag and drop)
  with a per-file result (indexed, skipped and why, or the error), a file list with
  chunk counts, and delete. Deleting a project (after a confirmation) deletes its
  chats, messages, files, chunks and copies.
- **RAG over uploads.** Ollama embeddings, SQLite + numpy, no extra server. Handles
  text and code, **PDFs** (text layer, with page markers) and **images**
  (described by an Ollama vision model). Retrieval uses the latest message plus a
  little recent conversation, so follow-ups like "and the second one?" work. The
  files an answer drew on are shown under it.
- **File transfer.** Upload from a phone or tablet straight into `uploads/` on the
  host, then download or delete from the same panel. This is separate from
  project files.
- **Streaming responses.** Tokens are streamed to the browser as they are
  generated, and the full reply is persisted once the stream completes.
- **Modern, mobile-friendly UI.** Served from `/`, no build step, no CDN (works
  offline on a LAN). Light and dark themes (follows the system, with a manual
  toggle that is remembered), markdown rendering with copyable code blocks,
  auto-growing input (Enter sends, Shift+Enter adds a line), toasts for errors
  such as Ollama being down, and a drawer sidebar on small screens. The theme is
  light = warm beige with Claude-terracotta accents, dark = near-black, all as
  tokens at the top of `static/style.css`; Netflix Sans is not bundled, so dark
  mode uses a Helvetica-style fallback unless it is installed.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- A pulled chat model (default is `qwen2.5-coder:7b`)
- For project files: an embedding model (`nomic-embed-text`)
- Optional, for images in projects: a vision model (default `qwen2.5vl:7b`,
  needs a recent Ollama)

## Setup

```bash
# 1. Pull a model
ollama pull qwen2.5-coder:7b

#    and an embedding model for project files
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
  database.py        SQLite connection helper, schema and in-place migration
  rag_store.py       the shared RagStore instance
  prompt.py          retrieval-query building and the chat history window
  ollama_errors.py   Ollama failures worded for people
  routes/
    sessions.py      chats and project sessions (list, create, delete, load)
    projects.py      project workspaces: create, list, detail, delete, new session
    project_files.py project files: multi-file upload (auto-indexed), list, delete, /rag query
    files.py         file transfer to/from uploads/
    chat.py          prompt assembly, streamed Ollama reply
    health.py        /api/health (Ollama reachability and installed models)
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
                                   ├─ server.db   projects, sessions, messages
                                   ├─ uploads/    files transferred from devices
                                   └─ rag.db + project_files/<id>/   RAG index + file copies
                                                                      /api/embed (vectors)
```

**Chat request (`POST /api/chat`)**

1. The user message is saved to `server.db`; a new chat is titled after its first
   message.
2. If the chat belongs to a project, a retrieval query is built from the latest
   message plus the recent *user* turns (a few hundred characters,
   `RAG_QUERY_HISTORY_CHARS`), embedded, and compared with the project's stored
   chunks. The top `RAG_TOP_K` are kept as long as their text fits
   `RAG_MAX_EXCERPT_CHARS`, and go in as a system message. A normal chat, an
   empty project, or any retrieval failure (Ollama down, stale index, database
   trouble, the 10 s query timeout) simply means no excerpts; chat never fails
   because retrieval did.
3. The history window is applied: the newest messages that fit
   `MAX_HISTORY_CHARS`, always including the new one, in their original order.
4. The prompt (excerpts, then history) is streamed to Ollama and tokens are relayed
   to the browser as they arrive. The reply is saved when the stream ends. The
   response header `X-RAG-Sources` lists the project files used (a JSON array).

**Upload (`POST /api/projects/{id}/files`)**

1. Each file is checked against its type's size cap while it is read (at most cap+1
   bytes are buffered). Starlette first spools the multipart body to a temp file, so a
   huge request still uses temp disk before the 413.
2. It is hashed together with `EMBED_MODEL` (images also with `VISION_MODEL`, so
   changing it re-describes them); an unchanged file is reported as `unchanged`.
3. Otherwise it is turned into text, split into ~1500-character chunks with 200
   overlap, and embedded through Ollama's `/api/embed`: text files as-is; PDFs per
   page with a `[page N]` marker at the start of every chunk; images via the vision
   model (asked to transcribe visible text and describe the content), each chunk
   starting with `[image: <name>]`.
4. Chunks and vectors go to `rag.db` (normalised float32 blobs). Only after
   embedding succeeds are the file's old chunks replaced and its copy written to
   `project_files/<id>/`, so a failure leaves the previous version intact.
5. The response has one entry per file, so one bad file does not hide the others.
   Uploading a name again replaces the earlier version. The web UI sends the files
   one request at a time so each shows its own progress.

Retrieval is a cosine search (a numpy dot product) over that project's chunks.

### Supported file types

| Type | Extensions | How it is indexed | Size cap |
| --- | --- | --- | --- |
| Text / code | anything that decodes as UTF-8 | UTF-8 text | 1 MB |
| PDF | `.pdf` | Text layer via `pypdf`, page by page | 20 MB, first 300 pages |
| Image | `.png` `.jpg` `.jpeg` `.webp` `.gif` `.bmp` | Described by `VISION_MODEL` | 10 MB |

Size is checked while the file is read. Per-file outcomes that are not errors:

| Result | Meaning |
| --- | --- |
| `indexed` / `captioned` | Stored and searchable (`captioned`: an image was described) |
| `unchanged` | Same content already indexed |
| `skipped` | Empty, binary, or over the size cap of the store |
| `skipped_pdf_no_text` | PDF without a text layer (a scan); there is no OCR for PDFs |
| `skipped_pdf_unreadable` | PDF is encrypted (password needed) or corrupt |
| `skipped_image_failed` | The vision model is missing or failed on this image; not remembered, so uploading again retries it |
| `skipped_image_disabled` | `VISION_MODEL` is empty |

A file that is too large for its type is an `error` entry with `status` 413; an
unsafe or unusable name (reserved Windows names, `:`, trailing dot, characters the
filesystem rejects, over 200 characters) is 400; Ollama or the
embedding model failing is 502. Skipped files are still listed (with no chunks) so
they can be seen and deleted.

`webp` and `bmp` support depends on your Ollama version; PNG/JPEG/GIF are the safe choice.
Indexing a large image means one vision call each, which can be slow on CPU. PDFs are
parsed with `pypdf` in a worker thread with no timeout, so a hostile PDF can tie up a
worker (size, page and text caps apply); fine for a LAN tool, but only upload PDFs you
trust. Excerpts shown to the model carry their file name, and `[page N]` /
`[image: ...]` markers where applicable.

## Configuration

Edit the constants in `backend/config.py`:

| Constant | Default | Purpose |
| --- | --- | --- |
| `OLLAMA` | `http://localhost:11434` | Where Ollama is listening |
| `MODEL` | `qwen2.5-coder:7b` | Model used for all chats |
| `MAX_HISTORY_CHARS` | `24000` | Newest history sent per chat (older turns stay stored, not sent) |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model used for RAG embeddings |
| `VISION_MODEL` | `qwen2.5vl:7b` | Ollama vision model for images; `""` disables image indexing |
| `VISION_TIMEOUT` | `300` | Seconds to wait for one image description |
| `QUERY_EMBED_TIMEOUT` | `10` | Seconds chat waits for the retrieval embedding before answering without it |
| `PDF_MAX_BYTES` | `20000000` | Largest PDF accepted |
| `IMAGE_MAX_BYTES` | `10000000` | Largest image accepted |
| `PDF_MAX_PAGES` | `300` | Pages read per PDF |
| `MAX_MESSAGE_CHARS` | `20000` | Longest chat message accepted (413 beyond it) |
| `CHAT_READ_TIMEOUT` | `300` | Seconds of silence from Ollama before a chat gives up |
| `HEALTH_TIMEOUT` / `HEALTH_CACHE_SECONDS` | `2` / `5` | `/api/health` asks Ollama with this timeout and caches the answer this long |
| `RAG_MIN_SCORE` | `0.5` | Minimum cosine similarity for an excerpt to be used (see below) |
| `RAG_TOP_K` | `5` | Chunks retrieved per question |
| `RAG_MAX_EXCERPT_CHARS` | `6000` | Total excerpt text injected per question |
| `RAG_QUERY_HISTORY_CHARS` | `400` | Earlier user turns added to the retrieval query |
| `RAG_QUERY_MAX_CHARS` | `2000` | Cap on the whole retrieval query |
| `RAG_CHUNK_CHARS` / `RAG_OVERLAP_CHARS` | `1500` / `200` | Chunk size and overlap |
| `MAX_UPLOAD_FILES` | `20` | Files accepted by one project upload request |
| `PROJECT_NAME_MAX` | `100` | Longest project name |

`RAG_MIN_SCORE` was chosen by measuring `nomic-embed-text` on short questions against
short paragraphs: on-topic chunks scored 0.59-0.79, off-topic ones 0.25-0.44, and a code
question against unrelated code 0.52. Longer chunks score lower, so if answers stop
finding your files, check real scores with `GET /api/projects/{id}/rag?q=...` and lower it;
re-measure if you change `EMBED_MODEL`.

Keep `MAX_HISTORY_CHARS` plus `RAG_MAX_EXCERPT_CHARS` inside the context length
configured in Ollama (roughly 3-4 characters per token), otherwise the prompt is
silently truncated by the model.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | `ok`, `model`, plus `status` (`ok`/`degraded`), `ollama` (reachable), `models` (`chat`, `embed`, `vision`: installed; `vision` is `null` when disabled), `max_message_chars`. Cached 5 s. A missing vision model does not degrade the status |
| `GET` | `/api/sessions` | List normal chats (no project), newest first |
| `POST` | `/api/sessions` | Create a chat; with `project_id` in the body, a session in that project |
| `GET` | `/api/sessions/{id}` | Session with full message history, `project_id` and `project_name` |
| `PUT` | `/api/sessions/{id}` | Rename: `{"title": ...}` (trimmed, 1-200 chars; 400 if blank, 404 if unknown); works inside projects |
| `DELETE` | `/api/sessions/{id}` | Delete a session and its messages |
| `GET` | `/api/projects` | List projects (`id`, `name`) |
| `POST` | `/api/projects` | Create a project from `{"name": ...}` (409 if the name exists, ignoring case and surrounding spaces) |
| `GET` | `/api/projects/{id}` | The project with its `sessions` and `files` |
| `DELETE` | `/api/projects/{id}` | Delete the project, its sessions, messages, files, chunks and copies |
| `POST` | `/api/projects/{id}/sessions` | Create a session inside the project |
| `POST` | `/api/projects/{id}/files` | Upload files (multipart field `files`, repeatable); each is indexed; per-file `results` |
| `GET` | `/api/projects/{id}/files` | Files with chunk counts |
| `DELETE` | `/api/projects/{id}/files?path=` | Remove a file's chunks and local copy |
| `GET` | `/api/projects/{id}/rag?q=&k=` | Query the project's index directly (debugging) |
| `POST` | `/api/chat` | Send a message, stream the reply. 400 if blank, 413 if over `MAX_MESSAGE_CHARS` (nothing is stored), 502 with a readable message if Ollama fails (the question is not kept). Headers: `X-RAG-Status` (`used`, `none`, `empty`, `failed`; absent in normal chats), `X-RAG-Sources` (files actually injected), `X-History-Trimmed` (older messages left out of the prompt, only when above 0) |
| `POST` | `/api/upload` | Upload a file to `uploads/` (file transfer) |
| `GET` | `/api/files` | List transferred files |
| `GET` | `/api/files/{name}` | Download a transferred file |
| `DELETE` | `/api/files/{name}` | Delete a transferred file |

Example:

```bash
curl -X POST http://localhost:8000/api/projects -H "Content-Type: application/json" \
  -d '{"name": "Thesis"}'                                    # {"id": 1, ...}
curl -F files=@notes.md -F files=@paper.pdf http://localhost:8000/api/projects/1/files
curl -X POST http://localhost:8000/api/projects/1/sessions   # {"id": 7}
curl -N http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": 7, "content": "summarise the paper"}'
```

## Data

All are created on first run in the repository root and are gitignored:

- `server.db` holds projects, sessions and messages
- `uploads/` holds transferred files
- `rag.db` holds chunks and embeddings; `project_files/<id>/` holds the local
  copy of every indexed file

Notes on behaviour:

- Upgrading from the folder-based version is automatic and non-destructive.
  `server.db` is migrated in place on start: old projects (which pointed at a disk
  folder) become empty workspaces with their name, their sessions stay in them,
  sessions pointing at a project that no longer exists become normal chats, and the
  old `path` and pinned-file `context` columns are kept but ignored. Chunks that a
  folder index left in `rag.db` still belong to their project: they are searched,
  listed as files, and removed by deleting the file or the project.
- Uploading a file name again replaces the earlier version.
- If you change `EMBED_MODEL` (or its vector size), old chunks are ignored and
  `/rag` answers 409; upload the files again to re-embed them. Chat keeps working,
  just without excerpts from those files.
- If Ollama or the embedding model is unavailable, chat still works without
  retrieved excerpts (reported as `failed`), and uploads report a 502 entry per file.
  Errors are worded for people: "The embedding model 'X' isn't installed. Run:
  ollama pull X", "Can't reach Ollama at <url>. Is it running?", "Ollama took too long
  to answer".
- A project chat that gets no excerpt scoring at least `RAG_MIN_SCORE` sends none, and
  the model is told so in a short system hint (also for an empty project or a failed
  search), so it can say the files don't cover the question. The UI shows under each
  answer: "Used project files: ...", "No project files matched this question", "This
  project has no searchable files yet" or "Could not search project files".
- If the reply fails before any text arrives, the question is removed again, so a reload
  does not show it unanswered and it is not sent as history; the UI offers Retry. Stop
  keeps the partial reply (stored once) and closes the Ollama request. While waiting for
  the first token (Ollama can be silent for a long time evaluating a prompt) the server
  watches for the browser leaving: if it does, the Ollama request is closed and the
  question removed, so Stop then Retry does not duplicate the exchange. The same cleanup
  runs if the response is discarded before streaming starts.
- If Ollama fails in the middle of a reply (error line, timeout, dropped connection), the
  text so far is kept and ends with `[Reply interrupted: <reason>]`, both on screen and in
  the stored message, so a cut-off answer is not mistaken for a complete one. A Stop by
  the user adds no such marker. (A connection that closes cleanly but early, with no
  error, cannot be told apart from a finished reply.)
- A new chat is named after its first sentence (up to 60 characters, cut at a word
  boundary with an ellipsis; a message of only punctuation uses that text) unless it was
  renamed first. Rename with the pencil button,
  the header button, or a double-click on the header title (Enter saves, Esc cancels).
- File names are UTF-8 end to end. A name such as `hÃ©llo` in a Windows
  console is the UTF-8 response read with the wrong codec, not a server problem.
- The UI shows a banner when Ollama is unreachable or the chat/embedding model is not
  installed (polled every 30 s), and a character counter near the message limit.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

Tests use a fake embedder and a fake captioner, and build their PDFs by hand, so
Ollama is not needed. The UI tests are static checks (ids, API routes used by
`app.js`, escaping); the JavaScript itself is not executed.

## Security

This server has **no authentication**. It is built for a trusted home network.
Before exposing it anywhere else, add an API key check or put a reverse proxy
with TLS in front of it. Uploaded filenames are flattened to their basename and
project file copies are confined to the project's own folder, but neither
substitutes for access control. Because there is no auth, any page open
in a browser on the same network can also call the write endpoints (CSRF);
a cross-site page can POST to the multipart upload endpoints but cannot read
any response; retrieved file excerpts are fed to the model as untrusted reference text.

## Licence

MIT
