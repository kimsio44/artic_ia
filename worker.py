"""Worker: consumes links from SQLite queue, fetches text and appends to input file."""
import argparse
import time
import sqlite3
from pathlib import Path
from link_queue import init_db, claim_next, push_links, mark_done, mark_failed, stats
from crawler import fetch_url, TextCrawlerHTMLParser, clean_extracted_text, normalize_url, same_site
# set up logging to print to console

def parse_args():
    p = argparse.ArgumentParser(description="Worker that processes queued links, extracts text and discovers new links.")
    p.add_argument("--db", default="data/links_queue.db", help="SQLite DB path")
    p.add_argument("--output", default="data/input.txt", help="File to append extracted text")
    p.add_argument("--delay", type=float, default=1.0, help="Delay between processing items when idle")
    p.add_argument("--max-depth", type=int, default=2, help="Maximum crawl depth for discovered links")
    p.add_argument("--same-domain", action="store_true", help="Only enqueue links from the same domain as the current page")
    p.add_argument("--max-retries", type=int, default=3, help="Max retries before marking failed")
    p.add_argument("--run-once", action="store_true", help="Process a single item then exit")
    return p.parse_args()

#save the extracted text to a file, with a header indicating the source URL
def append_text(path, url, text):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(f"### Source: {url}\n")
        f.write(text)
        f.write("\n\n")


def main():
    args = parse_args()
    conn = init_db(args.db)

    while True:
        try:
            item = claim_next(conn)
        except sqlite3.OperationalError as exc:
            if "database is locked" in str(exc):
                print(f"Database locked while claiming next item: {exc}")
                time.sleep(args.delay)
                continue
            raise

        if not item:
            print("Queue empty — sleeping...", flush=True)
            if args.run_once:
                break
            time.sleep(args.delay)
            continue

        _id = item["id"]
        url = item["url"]
        depth = item.get("depth", 0)
        retry = item.get("retry_count", 0)
        print(f"Processing {_id}: {url} (depth={depth} retry={retry})", flush=True)
        try:
            html = fetch_url(url)
            parser = TextCrawlerHTMLParser(url)
            parser.feed(html)
            text = clean_extracted_text(parser.get_text())
            if not text:
                raise ValueError("Texte vide après nettoyage")
            append_text(args.output, url, text)
            if depth < args.max_depth:
                discovered = []
                for link in parser.links:
                    normalized = normalize_url(link)
                    if not normalized:
                        continue
                    if args.same_domain and not same_site(url, normalized):
                        continue
                    discovered.append(normalized)
                pushed = push_links(conn, discovered, parent=url, depth=depth + 1)
                if pushed:
                    print(f"Enqueued {pushed} discovered links from {url}", flush=True)
            mark_done(conn, _id)
            print(f"Processed {_id} OK — appended to {args.output}", flush=True)
        except Exception as exc:
            err = str(exc)
            print(f"Error processing {_id}: {err}", flush=True)
            if retry + 1 >= args.max_retries:
                mark_failed(conn, _id, err)
                print(f"Marked {_id} failed (max retries)", flush=True)
            else:
                mark_failed(conn, _id, err)
                cur = conn.cursor()
                cur.execute("UPDATE links SET status='pending' WHERE id=? AND status='failed'", (_id,))
                conn.commit()
                print(f"Will retry {_id} later", flush=True)

        if args.run_once:
            break


if __name__ == "__main__":
    main()
