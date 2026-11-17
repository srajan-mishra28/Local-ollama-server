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


def init_db():
    with db() as c:
        # WAL lets a reader run while a write is in flight, which matters once
        # a phone and the laptop are both hitting this.
        c.execute("PRAGMA journal_mode = WAL")
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT NOT NULL,
                path      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT NOT NULL,
                project_id INTEGER,
                context    TEXT NOT NULL DEFAULT '[]',
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
