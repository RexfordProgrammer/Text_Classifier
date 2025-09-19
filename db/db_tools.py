# db_tools.py
import os
import sqlite3

# default DB file; can be overridden
_DB_FILE = os.environ.get("DB_FILE", "paragraphs_mistral.db")

def set_db_file(path: str) -> None:
    """Override the DB file path before calling init/insert/etc."""
    global _DB_FILE
    _DB_FILE = path

def _connect():
    return sqlite3.connect(_DB_FILE)

def init():
    """Create a single blocks table with type and content."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS blocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chapter_num INTEGER,
        block_num INTEGER,
        type TEXT CHECK(type IN ('header','paragraph')),
        content TEXT
    )
    """)
    conn.commit()
    conn.close()

def insert_block(chapter_num: int, block_num: int, type_: str, content: str) -> int:
    """Insert a header or paragraph into the blocks table."""
    if type_ not in ("header", "paragraph"):
        raise ValueError("type_ must be 'header' or 'paragraph'")
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO blocks (chapter_num, block_num, type, content) VALUES (?, ?, ?, ?)",
        (chapter_num, block_num, type_, content),
    )
    conn.commit()
    bid = cur.lastrowid
    conn.close()
    return bid

def has_block(content: str) -> bool:
    """Check if this content already exists (regardless of type)."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM blocks WHERE content = ? LIMIT 1", (content,))
    row = cur.fetchone()
    conn.close()
    return row is not None
