"""
clean_corpus.py — Nettoyage en streaming d'un fichier de corpus généré par crawler.py.

Pensé pour des fichiers volumineux (des centaines de Mo, voire plus) :
- Lit le fichier ligne par ligne / section par section, jamais en entier en RAM.
- Écrit le résultat au fur et à mesure dans un nouveau fichier (jamais de
  accumulation de tout le texte nettoyé en mémoire avant écriture).
- Réutilise les heuristiques déjà présentes dans crawler.py (is_junk_text_line)
  et ajoute des filtres utiles pour du contenu issu de wikis (Wikisource,
  Wikipédia) : numéros de page de scan, notes de bas de page isolées,
  répétitions de titres de section consécutifs.

Usage:
    python clean_corpus.py data/input.txt data/input.clean.txt
    python clean_corpus.py data/input.txt data/input.clean.txt --min-words 4
    python clean_corpus.py data/input.txt data/input.clean.txt --stats-only
"""

import argparse
import re
import sys
from pathlib import Path

# On réutilise les heuristiques déjà testées dans crawler.py plutôt que de
# les dupliquer/diverger.
from crawler import is_junk_text_line, MIN_CONTENT_WORDS  # noqa: F401

SOURCE_MARKER = "### Source:"

# --- Filtres additionnels spécifiques aux contenus de type wiki/scan ---

# Numéro de page de scan isolé sur sa propre ligne, ex: "12", "— 12 —", "[12]"
PAGE_NUMBER_RE = re.compile(r"^[\[\(\-—\s]*\d{1,4}[\]\)\-—\s]*$")

# Note de bas de page isolée, ex: "[1]", "[note 2]", "(1)"
FOOTNOTE_MARKER_RE = re.compile(r"^\[[a-zA-Z0-9 ]{1,12}\]$|^\(\d{1,3}\)$")

# Lignes de type "Chapitre X", "CHAPITRE PREMIER", souvent répétées en haut
# de chaque page scannée d'un même chapitre — on ne les supprime pas
# automatiquement (elles ont une valeur structurelle) mais on dédoublonne
# les répétitions consécutives identiques (déjà fait par clean_extracted_text
# au niveau ligne ; ici on étend la dédup à l'échelle d'un bloc plus large).


def is_scan_artifact(line):
    """Détecte les artefacts typiques de scan/OCR de livres numérisés
    (Wikisource), en plus du bruit générique déjà géré par crawler.py."""
    stripped = line.strip()
    if not stripped:
        return False
    if PAGE_NUMBER_RE.match(stripped):
        return True
    if FOOTNOTE_MARKER_RE.match(stripped):
        return True
    return False


def clean_block_streaming(lines_iter, min_words=MIN_CONTENT_WORDS):
    """Nettoie un bloc de lignes (générateur), en conservant la logique de
    crawler.is_junk_text_line tout en ajoutant le filtre scan/OCR.
    Retourne un générateur de lignes nettoyées (pas de liste en mémoire)."""
    prev_line = None
    blank_pending = False

    for raw_line in lines_iter:
        line = raw_line.rstrip("\n")
        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            if prev_line is not None:  # éviter plusieurs lignes vides de suite
                blank_pending = True
            continue

        if is_junk_text_line(line) or is_scan_artifact(line):
            continue

        if line == prev_line:  # dédoublonnage de lignes consécutives identiques
            continue

        if blank_pending:
            yield ""
            blank_pending = False

        yield line
        prev_line = line


def stream_clean_file(input_path, output_path, min_words=MIN_CONTENT_WORDS, stats_only=False):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_sources = 0
    total_lines_in = 0
    total_lines_out = 0

    out_handle = None if stats_only else output_path.open("w", encoding="utf-8")

    def emit(text):
        if out_handle is not None:
            out_handle.write(text)

    try:
        current_header = None
        current_block_lines = []

        def flush_block():
            nonlocal total_sources, total_lines_in, total_lines_out
            if current_header is None:
                return
            total_sources += 1
            total_lines_in += len(current_block_lines)
            cleaned = list(clean_block_streaming(iter(current_block_lines), min_words=min_words))
            total_lines_out += len(cleaned)
            if cleaned and any(l.strip() for l in cleaned):
                emit(current_header + "\n")
                emit("\n".join(cleaned).strip() + "\n\n")

        with input_path.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                if raw_line.startswith(SOURCE_MARKER):
                    flush_block()
                    current_header = raw_line.rstrip("\n")
                    current_block_lines = []
                else:
                    current_block_lines.append(raw_line)
            flush_block()  # dernier bloc du fichier
    finally:
        if out_handle is not None:
            out_handle.close()

    return {
        "sources": total_sources,
        "lines_in": total_lines_in,
        "lines_out": total_lines_out,
        "lines_removed": total_lines_in - total_lines_out,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nettoie en streaming un fichier corpus produit par crawler.py (gère les très gros fichiers sans tout charger en RAM)."
    )
    parser.add_argument("input_file", help="Fichier source (ex: data/input.txt)")
    parser.add_argument("output_file", nargs="?", default=None,
                         help="Fichier de sortie nettoyé. Si omis, écrit à côté avec suffixe .clean.txt")
    parser.add_argument("--min-words", type=int, default=MIN_CONTENT_WORDS,
                         help="Nombre minimum de mots pour garder une ligne comme contenu valide")
    parser.add_argument("--stats-only", action="store_true",
                         help="Ne pas écrire de fichier, juste afficher les statistiques de nettoyage")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Fichier introuvable : {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output_file
    if output_path is None:
        output_path = str(input_path.with_suffix("")) + ".clean.txt"

    print(f"Nettoyage en streaming de {input_path} -> {output_path if not args.stats_only else '(stats uniquement)'}")
    stats = stream_clean_file(input_path, output_path, min_words=args.min_words, stats_only=args.stats_only)

    print()
    print("=== Statistiques ===")
    print(f"Sources traitées      : {stats['sources']}")
    print(f"Lignes en entrée       : {stats['lines_in']}")
    print(f"Lignes en sortie       : {stats['lines_out']}")
    print(f"Lignes supprimées      : {stats['lines_removed']} "
          f"({100 * stats['lines_removed'] / max(1, stats['lines_in']):.1f}%)")
    if not args.stats_only:
        print(f"\nFichier nettoyé écrit dans : {output_path}")


if __name__ == "__main__":
    main()