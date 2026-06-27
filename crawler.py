import argparse
import gzip
import json
import html
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import deque
from html.parser import HTMLParser
from pathlib import Path

CLICKABLE_TAGS = {"a", "area", "link", "iframe"}
IGNORED_SCHEMES = {"mailto", "tel", "javascript", "data", "news", "ftp"}
RESOURCE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".mp4", ".mp3",
    ".pdf", ".zip", ".rar", ".exe", ".dmg", ".iso", ".css", ".js",
    ".xml", ".json", ".rss", ".txt", ".ico", ".psd", ".csv", ".doc", ".docx",
    ".xls", ".xlsx", ".ppt", ".pptx", ".webmanifest", ".map", ".woff", ".woff2", ".ttf", ".eot",
}

NON_HTML_PATH_SEGMENTS = {"rss", "atom", "feed"}

# Chemins/scripts non-HTML à exclure explicitement (ex: l'API MediaWiki,
# qui ne porte pas d'extension de fichier et passerait sinon le filtre).
EXCLUDED_PATH_SUBSTRINGS = (
    "/w/api.php",
    "/w/index.php",
    "/wiki/special:",
)

# Préfixes de namespace MediaWiki à exclure quand on veut du texte
# d'articles "normaux" plutôt que des pages méta du wiki.
# Comparaison insensible à la casse sur la partie après "/wiki/".
EXCLUDED_WIKI_NAMESPACES = (
    "wikipédia:", "wikipedia:",
    "fichier:", "file:",
    "catégorie:", "category:",
    "modèle:", "template:",
    "aide:", "help:",
    "portail:", "portal:",
    "discussion:", "talk:",
    "spécial:", "special:",
    "utilisateur:", "user:",
)

BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "nav", "aside",
    "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "tr", "td", "th",
    "blockquote", "pre", "address", "figure", "figcaption",
}


def encode_url(url):
    """Encode proprement une URL potentiellement non-ASCII (accents, etc.)
    en conservant les caractères déjà valides pour une URL.

    C'est la fonction qui corrige le crash
    'ascii' codec can't encode character '\\xe9' :
    sans elle, des hrefs comme '/wiki/Acide_férulique' restent avec des
    caractères non-ASCII bruts, et urllib plante en tentant de les envoyer
    tels quels dans la requête HTTP.
    """
    parsed = urllib.parse.urlsplit(url)
    # safe="/%" : on ne touche pas aux '/' de structure, et on ne ré-encode
    # pas les '%' déjà présents (cas où l'URL est déjà partiellement encodée).
    path = urllib.parse.quote(parsed.path, safe="/%:@")
    query = urllib.parse.quote(parsed.query, safe="=&%:@/")
    netloc = parsed.netloc.encode("idna").decode("ascii") if parsed.netloc else parsed.netloc
    return urllib.parse.urlunsplit((parsed.scheme, netloc, path, query, ""))


def is_html_candidate(parsed_url):
    if parsed_url.scheme not in {"http", "https"}:
        return False
    if parsed_url.path and Path(parsed_url.path).suffix.lower() in RESOURCE_EXTENSIONS:
        return False
    last_segment = Path(parsed_url.path.rstrip("/")).name.lower()
    if last_segment in NON_HTML_PATH_SEGMENTS:
        return False

    full_lower = parsed_url.geturl().lower()
    if any(substr in full_lower for substr in EXCLUDED_PATH_SUBSTRINGS):
        return False

    return True


def is_excluded_namespace(parsed_url):
    """Filtre les pages méta des wikis MediaWiki (Wikipédia:, Fichier:, ...)
    pour ne garder que les vrais articles."""
    path = parsed_url.path
    marker = "/wiki/"
    idx = path.lower().find(marker)
    if idx == -1:
        return False
    after = path[idx + len(marker):]
    after_decoded = urllib.parse.unquote(after).lower()
    return any(after_decoded.startswith(ns) for ns in EXCLUDED_WIKI_NAMESPACES)


class TextCrawlerHTMLParser(HTMLParser):
    def __init__(self, base_url, allowed_domain=None):
        super().__init__(convert_charrefs=False)
        self.base_url = base_url
        self.base_href = base_url
        self.allowed_domain = allowed_domain
        self._text_parts = []
        self.links = set()
        self._skip_content = False
        self._skip_tag_stack = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "base" and attrs.get("href"):
            self.base_href = urllib.parse.urljoin(self.base_url, attrs["href"])
        if tag in {"script", "style", "noscript"}:
            self._skip_content = True
            self._skip_tag_stack.append(tag)
        if tag in CLICKABLE_TAGS:
            url = attrs.get("href") or attrs.get("src")
            if url:
                normalized = self._normalize_url(url)
                if normalized:
                    self.links.add(normalized)
        if tag in BLOCK_TAGS:
            self._text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self._skip_tag_stack:
            self._skip_tag_stack.pop()
            self._skip_content = bool(self._skip_tag_stack)
        if tag in BLOCK_TAGS:
            self._text_parts.append("\n")

    def handle_data(self, data):
        if self._skip_content:
            return
        text = data.strip()
        if text:
            self._text_parts.append(text)

    def handle_entityref(self, name):
        if not self._skip_content:
            self._text_parts.append(html.unescape(f"&{name};"))

    def handle_charref(self, name):
        if not self._skip_content:
            self._text_parts.append(html.unescape(f"&#{name};"))

    def get_text(self):
        text = "".join(self._text_parts)
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t\f\r]+", " ", text)
        text = re.sub(r" *\n+ *", "\n\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _normalize_url(self, url):
        url = urllib.parse.urljoin(self.base_href, url)
        url = encode_url(url)
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme in IGNORED_SCHEMES:
            return None
        if not is_html_candidate(parsed):
            return None
        if is_excluded_namespace(parsed):
            return None
        if self.allowed_domain and parsed.netloc.lower() != self.allowed_domain:
            return None
        return parsed._replace(fragment="").geturl()


def fetch_url(url, timeout=15, max_retries=2, retry_delay=5.0):
    url = encode_url(url)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; simple-text-crawler/1.0; +https://example.com)",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                if not any(ct in content_type for ct in ("text/html", "application/xhtml+xml")):
                    raise ValueError(f"Page non-HTML : {content_type}")

                raw = response.read()
                encoding = response.headers.get("Content-Encoding", "").lower()
                if encoding == "gzip":
                    raw = gzip.decompress(raw)
                elif encoding == "deflate":
                    try:
                        raw = zlib.decompress(raw)
                    except zlib.error:
                        raw = zlib.decompress(raw, -zlib.MAX_WBITS)

                charset = response.headers.get_content_charset(failobj="utf-8")
                return raw.decode(charset, errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 502, 503, 504} and attempt < max_retries:
                wait = retry_delay
                retry_after = exc.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = float(retry_after)
                time.sleep(wait)
                continue
            raise

    raise ValueError(f"Échec de récupération après {max_retries} tentatives : {url}")


def collect_links(seed_urls, max_pages=50, max_depth=2, same_domain=False, delay=1.0):
    visited = set()
    scheduled = set()
    queue = deque()
    found_links = set()

    # Domaine de référence pour le filtrage same-domain : celui de la
    # première URL de départ. Toutes les seeds doivent normalement être
    # sur le même site si on veut un filtrage cohérent.
    allowed_domain = None

    for url in seed_urls:
        normalized = normalize_url(url, enforce_namespace_filter=False)
        if normalized:
            if allowed_domain is None:
                allowed_domain = urllib.parse.urlparse(normalized).netloc.lower()
            queue.append((normalized, 0))
            scheduled.add(normalized)

    domain_filter = allowed_domain if same_domain else None

    while queue and len(visited) < max_pages:
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue

        print(f"Collecting ({len(visited)+1}/{max_pages}): {url}")
        try:
            html_content = fetch_url(url)
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            print(f"  Impossible de charger {url}: {exc}")
            visited.add(url)
            continue

        parser = TextCrawlerHTMLParser(url, allowed_domain=domain_filter)
        parser.feed(html_content)
        visited.add(url)

        for link in parser.links:
            if not link or link in visited:
                continue
            if same_domain and not same_site(url, link):
                continue
            found_links.add(link)
            if link not in scheduled and depth < max_depth and len(visited) + len(queue) < max_pages:
                queue.append((link, depth + 1))
                scheduled.add(link)

        time.sleep(delay)

    return sorted(found_links)


def analyze_links(links, output_file, overwrite=False, delay=1.0):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    successful_links = []

    with output_path.open(mode, encoding="utf-8") as f:
        for index, url in enumerate(links, start=1):
            print(f"Analyse {index}/{len(links)}: {url}")
            try:
                html_content = fetch_url(url)
            except (urllib.error.URLError, ValueError, TimeoutError) as exc:
                print(f"  Échec {url}: {exc}")
                continue

            parser = TextCrawlerHTMLParser(url)
            parser.feed(html_content)
            page_text = clean_extracted_text(parser.get_text())
            if not page_text:
                continue

            f.write(f"### Source: {url}\n")
            f.write(page_text + "\n\n")
            successful_links.append(url)
            time.sleep(delay)

    print(f"Texte ajouté dans {output_path}")
    return successful_links


# Préfixes génériques de bruit de navigation/footer, indépendants du site.
JUNK_PREFIXES = (
    "skip to content",
    "back to top",
    "copyright ©",
    "all rights reserved",
)

# Nombre minimum de mots pour qu'une ligne soit considérée comme du contenu
# (en dessous, c'est presque toujours un libellé de menu/bouton/lien isolé).
MIN_CONTENT_WORDS = 4


def is_junk_text_line(line):
    """Heuristique générique : repère le bruit de navigation/UI sans dépendre
    du vocabulaire d'un site en particulier."""
    line = line.strip()
    if not line:
        return False
    lower_line = line.lower()

    if any(lower_line.startswith(prefix) for prefix in JUNK_PREFIXES):
        return True

    # Ligne composée uniquement de ponctuation/symboles (séparateurs, puces…)
    if re.fullmatch(r"[^\w\s]{3,}", line):
        return True

    # Ligne très courte sans aucun caractère alphanumérique (icônes, séparateurs)
    if len(line) <= 3 and not any(ch.isalnum() for ch in line):
        return True

    words = re.findall(r"\w+", lower_line)
    # Trop peu de mots pour être une vraie phrase : probablement un libellé
    # de navigation ("Docs", "Get Started", "Read more"...). On garde quand
    # même les lignes qui se terminent par une ponctuation de phrase, car
    # une courte phrase complète ("Ça marche.") reste du contenu valide.
    if len(words) < MIN_CONTENT_WORDS and not line.rstrip().endswith((".", "!", "?", "…")):
        return True

    return False


def clean_extracted_text(text):
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"\s+", " ", line)
        if not line:
            if lines and lines[-1] == "":
                continue
            lines.append("")
            continue
        if is_junk_text_line(line):
            continue
        if lines and line == lines[-1]:
            continue
        lines.append(line)

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def clean_data_file(input_file):
    path = Path(input_file)
    if not path.exists():
        print(f"Fichier introuvable : {path}")
        return

    content = path.read_text(encoding="utf-8")
    sections = content.split("### Source:")
    if len(sections) <= 1:
        print(f"Aucune section source trouvée dans {path}")
        return

    cleaned_sections = [sections[0].rstrip()]
    for section in sections[1:]:
        header, *body_lines = section.splitlines()
        block_text = "\n".join(body_lines).strip()
        cleaned_text = clean_extracted_text(block_text)
        cleaned_sections.append(f"### Source:{header.strip()}\n{cleaned_text}" if cleaned_text else f"### Source:{header.strip()}")

    cleaned_content = "\n\n".join([sec for sec in cleaned_sections if sec.strip()]) + "\n"
    path.write_text(cleaned_content, encoding="utf-8")
    print(f"Fichier nettoyé : {path}")


def save_links_json(links, output_file, seeds=None):
    """Sauvegarde la liste des liens traités avec succès dans un fichier JSON."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": seeds or [],
        "count": len(links),
        "links": links,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Liens sauvegardés dans {output_path} ({len(links)} liens)")


def normalize_url(url, enforce_namespace_filter=False):
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    url = encode_url(url)
    parsed = urllib.parse.urlparse(url)
    if not is_html_candidate(parsed):
        return None
    # Le filtre de namespace (Wikipédia:, Fichier:, ...) ne s'applique
    # qu'aux liens découverts pendant le crawl, pas aux seeds fournies
    # explicitement par l'utilisateur : si on lui donne volontairement
    # une URL "Wikipédia:Accueil_principal" comme point de départ, on la
    # respecte plutôt que de la rejeter silencieusement.
    if enforce_namespace_filter and is_excluded_namespace(parsed):
        return None
    return parsed._replace(fragment="").geturl()


def same_site(base_url, other_url):
    return urllib.parse.urlparse(base_url).netloc.lower() == urllib.parse.urlparse(other_url).netloc.lower()


def parse_args():
    parser = argparse.ArgumentParser(description="Crawler et analyseur de site pour récupérer du texte dans data/input.txt.")
    parser.add_argument("seed_urls", nargs="+", help="URL de départ ou fichiers contenant des URL à crawler")
    parser.add_argument("--max-pages", type=int, default=50, help="Nombre maximum de pages à visiter pour collecter les liens")
    parser.add_argument("--max-depth", type=int, default=2, help="Profondeur pour suivre les liens")
    parser.add_argument("--same-domain", action="store_true", help="Ne suivre que les liens du même domaine")
    parser.add_argument("--delay", type=float, default=1.0, help="Délai en secondes entre chaque requête")
    parser.add_argument("--links-json", type=str, default="data/links.json", help="Fichier JSON pour stocker les liens collectés")
    parser.add_argument("--save-to-input", type=str, default="data/input.txt", help="Fichier dans lequel écrire le texte extrait de chaque lien")
    parser.add_argument("--overwrite-input", action="store_true", help="Réécrire data/input.txt au lieu d'ajouter")
    parser.add_argument("--clean-input", action="store_true", help="Nettoyer le fichier de texte extrait après génération")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = []
    for value in args.seed_urls:
        if Path(value).is_file():
            seeds.extend([line.strip() for line in Path(value).read_text(encoding="utf-8").splitlines() if line.strip()])
        else:
            seeds.append(value)

    links = collect_links(
        seeds,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        same_domain=args.same_domain,
        delay=args.delay,
    )
    successful_links = analyze_links(links, args.save_to_input, overwrite=args.overwrite_input, delay=args.delay)
    if args.clean_input:
        clean_data_file(args.save_to_input)
    save_links_json(successful_links, args.links_json, seeds)


if __name__ == "__main__":
    main()