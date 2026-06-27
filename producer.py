"""Producer: discovers links and pushes them into SQLite queue."""
import argparse
import time
from link_queue import init_db, push_links
from crawler import collect_links, normalize_url


def parse_args():
    p = argparse.ArgumentParser(description="Seed the URL queue for exponential crawling.")
    p.add_argument("seed_urls", nargs="+", help="Seed URL(s) or files with URLs")
    p.add_argument("--db", default="data/links_queue.db", help="SQLite DB path")
    p.add_argument("--max-pages", type=int, default=100, help="Max pages per discovery batch")
    p.add_argument("--max-depth", type=int, default=1, help="Max depth to follow during seeding")
    p.add_argument("--same-domain", action="store_true")
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--interval", type=float, default=0.0, help="If >0 run discovery in loop every N seconds")
    p.add_argument("--seed-only", action="store_true", help="Only enqueue seed URLs without crawling first")
    return p.parse_args()


def main():
    args = parse_args()
    conn = init_db(args.db)

    seeds = []
    for v in args.seed_urls:
        seeds.append(v)

    if args.seed_only:
        normalized = [normalize_url(u) for u in seeds]
        normalized = [u for u in normalized if u]
        n = push_links(conn, normalized, parent=None, depth=0)
        print(f"Seeded {len(normalized)} URLs into queue")
        return

    while True:
        print("Running discover pass...")
        links = collect_links(
            seeds,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            same_domain=args.same_domain,
            delay=args.delay,
        )
        normalized = [normalize_url(u) for u in links]
        normalized = [u for u in normalized if u]
        n = push_links(conn, normalized)
        print(f"Discovered {len(links)} links, pushed {n} new into queue")
        if args.interval <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
