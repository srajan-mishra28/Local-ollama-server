# Local Ollama Server

A single-file FastAPI gateway in front of a locally running [Ollama](https://ollama.com)
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
- **File transfer.** Upload from a phone or tablet straight into `uploads/` on the
  host, then download or delete from the same panel.
- **Streaming responses.** Tokens are streamed to the browser as they are
  generated, and the full reply is persisted once the stream completes.
- **Mobile-friendly UI.** Served from `/`, no build step, no dependencies beyond
  the Python packages below.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- A pulled model (default is `qwen2.5-coder:7b`)

## Setup

```bash
# 1. Pull a model
ollama pull qwen2.5-coder:7b

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

## Configuration

Edit the constants at the top of `app.py`:

| Constant | Default | Purpose |
| --- | --- | --- |
| `OLLAMA` | `http://localhost:11434` | Where Ollama is listening |
| `MODEL` | `qwen2.5-coder:7b` | Model used for all sessions |
| `MAX_CONTEXT_BYTES` | `120000` | Cap on injected project context |
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

Both are created on first run next to `app.py` and are gitignored:

- `server.db` holds projects, sessions and messages
- `uploads/` holds transferred files

## Security

This server has **no authentication**. It is built for a trusted home network.
Before exposing it anywhere else, add an API key check or put a reverse proxy
with TLS in front of it. Uploaded filenames are flattened to their basename and
project context reads are confined to the registered project root, but neither
substitutes for access control.

## Licence

MIT
