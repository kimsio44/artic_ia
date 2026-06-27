"""
front_dialogue.py — Télécharge le sous-ensemble "Claire-fr" (dialogue
français) du Lucie Training Dataset (OpenLLM-France), et l'AJOUTE à
data/input.txt en complément du corpus Wikipédia déjà présent.

✅ Dataset PUBLIC, format Parquet (pas de script de chargement custom,
donc compatible avec les versions récentes de `datasets`), SANS
authentification requise.

Le sous-ensemble Claire-fr est justement la même donnée dialogue/théâtre
que Claire-Dialogue-French-0.1 (gated), mais repackagée ici dans un
dataset plus large qui lui n'est pas gated.

Licence CC BY-NC-SA 4.0 — usage non-commercial uniquement.
"""

import os
from datasets import load_dataset

OUTPUT_PATH = "data/input.txt"
MAX_DOCS = 20_000     # ajustez selon le volume voulu
PRINT_EVERY = 500


def main():
    print("⏳ Chargement du sous-ensemble Claire-fr (Lucie-Training-Dataset)...")
    print("   (dataset public, format Parquet, pas d'authentification nécessaire)")

    ds = load_dataset(
        "OpenLLM-France/Lucie-Training-Dataset",
        "Claire-fr",
        split="train",
        streaming=True,
    )

    # Diagnostic rapide avant d'écrire quoi que ce soit.
    print("🔍 Tentative de récupération du premier échantillon...")
    first = next(iter(ds), None)
    if first is None:
        print("❌ Aucun échantillon récupéré — l'itérateur est vide.")
        return
    print(f"🔍 Clés disponibles : {list(first.keys())}")
    print(f"🔍 Aperçu du texte : {first.get('text', '')[:200]!r}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    written = 0
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        # On réinjecte le premier échantillon déjà consommé par le diagnostic.
        for i, sample in enumerate([first, *ds]):
            text = (sample.get("text") or "").strip()
            if not text:
                continue

            doc_id = sample.get("id", i)
            f.write(f"### Source: lucie-claire-fr-{doc_id}\n")
            f.write(text + "\n\n")
            f.flush()
            os.fsync(f.fileno())
            written += 1

            if written % PRINT_EVERY == 0:
                print(f"  ... {written} documents ajoutés")

            if written >= MAX_DOCS:
                break

    print(f"✅ Terminé — {written} documents ajoutés à {OUTPUT_PATH}")
    print("⚠️  Licence CC BY-NC-SA 4.0 : usage non-commercial uniquement.")


if __name__ == "__main__":
    main()