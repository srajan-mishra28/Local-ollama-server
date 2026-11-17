"""SQLite access and schema for projects, sessions and messages."""

import sqlite3
from contextlib import contextmanager

from backend.config import DB_PATH


@contextmanager
def db():
    """Commit on clean exit, always close.

    sqlite3's own connection context manager commits but does NOT close, so
    using it directly leaks a file handle per request.
    """
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _has_column(c: sqlite3.Connection, table: str, column: str) -> bool:
    return any(r["name"] == column for r in c.execute(f"PRAGMA table_info({table})"))


def init_db():
    """Create the schema, and bring an older server.db up to date in place.

    Idempotent and non-destructive: nothing is dropped. Older databases differ
    in that ``projects`` was a registered disk folder (``path``) and sessions
    carried a pinned-file list (``context``). Both columns are kept but unused:
    a project is now just a named workspace, so old folder projects become
    empty workspaces with their name, and their sessions stay inside them.
    """
    with db() as c:
        # WAL lets a reader run while a write is in flight, which matters once
        # a phone and the laptop are both hitting this.
        c.execute("PRAGMA journal_mode = WAL")
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT NOT NULL,
                path      TEXT NOT NULL DEFAULT ''  -- legacy (folder), unused; always ''
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT NOT NULL,
                project_id INTEGER,                 -- NULL: a normal chat
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id);
            """
        )
        if not _has_column(c, "sessions", "project_id"):
            c.execute("ALTER TABLE sessions ADD COLUMN project_id INTEGER")
        c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id)")
        # Old sessions could point at a project that was since deleted. They
        # would belong to nothing the UI lists, so they become normal chats.
        c.execute(
            "UPDATE sessions SET project_id = NULL "
            "WHERE project_id IS NOT NULL AND project_id NOT IN (SELECT id FROM projects)"
        )
