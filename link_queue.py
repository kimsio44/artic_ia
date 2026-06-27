import sqlite3
import time
from pathlib import Path


def _retry_db_action(action, retries=5, delay=0.5):
    for attempt in range(retries):
        try:
            return action()
        except sqlite3.OperationalError as exc:
            if "database is locked" in str(exc) and attempt < retries - 1:
                time.sleep(delay)
                continue
            raise


def init_db(db_path="data/links_queue.db"):
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            status TEXT DEFAULT 'pending',
            retry_count INTEGER DEFAULT 0,
            discovered_at REAL,
            processed_at REAL,
            parent TEXT,
            depth INTEGER DEFAULT 0,
            last_error TEXT
        )
        """
    )
    conn.commit()
    return conn


def push_links(conn, urls, parent=None, depth=0):
    cur = conn.cursor()
    now = time.time()
    inserted = 0
    for u in urls:
        try:
            def action():
                cur.execute(
                    "INSERT INTO links (url, discovered_at, parent, depth) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(url) DO UPDATE SET depth=excluded.depth, parent=excluded.parent "
                    "WHERE excluded.depth < depth",
                    (u, now, parent, depth),
                )
            _retry_db_action(action)
            if cur.rowcount:
                inserted += 1
        except Exception:
            continue
    conn.commit()
    return inserted


def claim_next(conn):
    cur = conn.cursor()
    def action():
        cur.execute(
            "UPDATE links SET status='processing' "
            "WHERE id = (SELECT id FROM links WHERE status='pending' ORDER BY depth ASC, discovered_at ASC LIMIT 1) "
            "AND status='pending' "
            "RETURNING id, url, retry_count, depth, parent"
        )
        return cur.fetchone()
    try:
        row = _retry_db_action(action)
        conn.commit()
    except sqlite3.OperationalError:
        conn.rollback()
        raise
    if not row:
        return None
    _id, url, retry, depth, parent = row
    return {"id": _id, "url": url, "retry_count": retry, "depth": depth, "parent": parent}


def mark_done(conn, _id):
    cur = conn.cursor()
    def action():
        cur.execute("UPDATE links SET status='done', processed_at=? WHERE id=?", (time.time(), _id))
    _retry_db_action(action)
    conn.commit()


def mark_failed(conn, _id, err_msg=None):
    cur = conn.cursor()
    def action():
        cur.execute(
            "UPDATE links SET status='failed', retry_count=retry_count+1, last_error=? WHERE id=?",
            (err_msg or "", _id),
        )
    _retry_db_action(action)
    conn.commit()


def stats(conn):
    cur = conn.cursor()
    cur.execute("SELECT status, COUNT(*) FROM links GROUP BY status")
    return dict(cur.fetchall())
