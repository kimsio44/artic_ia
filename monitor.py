"""Monitor the SQLite link queue and active worker processes."""
import argparse
import sqlite3
import subprocess
import time
from pathlib import Path


def init_db(db_path="data/links_queue.db"):
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database file not found: {path}")
    return sqlite3.connect(str(path), timeout=30)


def get_stats(conn):
    cur = conn.cursor()
    cur.execute("SELECT status, COUNT(*) FROM links GROUP BY status")
    counts = dict(cur.fetchall())
    cur.execute("SELECT COUNT(*) FROM links")
    total = cur.fetchone()[0]
    cur.execute("SELECT url FROM links WHERE status='pending' ORDER BY discovered_at LIMIT 5")
    pending = [row[0] for row in cur.fetchall()]
    cur.execute("SELECT url FROM links WHERE status='processing' ORDER BY discovered_at LIMIT 5")
    processing = [row[0] for row in cur.fetchall()]
    return counts, total, pending, processing


def count_workers():
    try:
        result = subprocess.run(["pgrep", "-fc", "worker.py"], capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(description="Monitor the crawling queue and worker activity.")
    p.add_argument("--db", default="data/links_queue.db", help="SQLite queue database")
    p.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    p.add_argument("--show-urls", action="store_true", help="Show sample pending and processing URLs")
    return p.parse_args()


def main():
    args = parse_args()
    conn = init_db(args.db)

    while True:
        counts, total, pending, processing = get_stats(conn)
        active_workers = count_workers()
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        pending_count = counts.get("pending", 0)
        processing_count = counts.get("processing", 0)
        done_count = counts.get("done", 0)
        failed_count = counts.get("failed", 0)

        print("=" * 80)
        print(f"{now} | total={total} | pending={pending_count} | processing={processing_count} | done={done_count} | failed={failed_count}")
        if active_workers is not None:
            print(f"Active worker processes: {active_workers}")
        if args.show_urls:
            if pending:
                print("\nPending URLs:")
                for url in pending:
                    print(f" - {url}")
            if processing:
                print("\nProcessing URLs:")
                for url in processing:
                    print(f" - {url}")
        print("=" * 80)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
