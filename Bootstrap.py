"""
bootstrap.py — Script d'installation automatique pour les volontaires
de l'entraînement décentralisé.

Ce script :
1. Vérifie/installe les dépendances Python nécessaires (pip install)
2. Clone (ou met à jour) le code du projet depuis GitHub
3. Télécharge un extrait de Wikipédia FR pour servir de dataset local
4. Lance train_hivemind.py en rejoignant automatiquement le réseau

⚠️ REMPLACER avant utilisation :
- GITHUB_REPO_URL : l'URL de votre repo public
- INITIAL_PEER_ADDR : l'adresse de votre pair initial (IP publique + port + ID)
"""

import os
import sys
import subprocess
import shutil

# ⚠️ À REMPLACER avec vos vraies valeurs ───────────────────────────────
GITHUB_REPO_URL = "https://github.com/kimsio44/artic_ia.git"
INITIAL_PEER_ADDR = "/ip4/91.163.75.214/tcp/14000/p2p/12D3KooWNBUdXz5XSvTzjQTuVhSNx8WjwJSonmpqYkeM2WXHXFm4"
# ────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.join(os.path.expanduser("~"), "gpt_fr_decentralized")
REQUIRED_PACKAGES = ["torch", "tiktoken", "hivemind", "datasets"]
WIKI_ARTICLES_COUNT = 500  # volume raisonnable pour un pair volontaire


def run(cmd, **kwargs):
    print(f"   $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def check_python():
    # Note : dans un .exe packagé avec PyInstaller, Python est déjà
    # embarqué — cette vérification reste utile si bootstrap.py est lancé
    # directement en .py (ex: par vous pour tester), mais n'a plus
    # vraiment de sens dans le .exe final puisque la version est fixée au
    # moment du packaging.
    print(" Vérification de Python...")
    version = sys.version_info
    if version < (3, 9):
        print(f" Python {version.major}.{version.minor} détecté — version 3.9+ requise.")
        print("   Installez Python depuis https://www.python.org/downloads/ puis relancez.")
        sys.exit(1)
    print(f" Python {version.major}.{version.minor}.{version.micro} OK")


def check_git():
    print("🔍 Vérification de Git...")
    if shutil.which("git") is None:
        print(" Git n'est pas installé ou n'est pas dans le PATH.")
        print("   Téléchargez-le ici puis relancez ce programme :")
        print("   👉 https://git-scm.com/downloads")
        sys.exit(1)
    print("✅ Git trouvé")


def get_python_command():
    """Retourne la commande Python à utiliser pour pip install / lancer
    train.py. Dans un .exe figé par PyInstaller, sys.executable pointe
    vers le .exe lui-même (pas un interpréteur Python utilisable pour
    'pip install' ou exécuter un script .py externe) — il faut alors
    chercher un Python système séparé sur la machine du volontaire."""
    if getattr(sys, "frozen", False):
        for candidate in ("python3", "python"):
            found = shutil.which(candidate)
            if found:
                return found
        print("❌ Aucun interpréteur Python système trouvé sur cette machine.")
        print("   Installez Python depuis https://www.python.org/downloads/")
        print("   (cochez bien 'Add Python to PATH' pendant l'installation), puis relancez.")
        sys.exit(1)
    return sys.executable


PYTHON_CMD = None  # initialisé dans main(), après check_python/check_git


def install_dependencies():
    print("\n📦 Installation des dépendances...")
    for package in REQUIRED_PACKAGES:
        print(f"   ⏳ Installation de {package}...")
        run([PYTHON_CMD, "-m", "pip", "install", package, "--quiet"])


def setup_project_code():
    print(f"\n📂 Récupération du code du projet dans {PROJECT_DIR}...")
    if os.path.exists(PROJECT_DIR):
        print("   Dossier existant détecté, mise à jour (git pull)...")
        try:
            run(["git", "-C", PROJECT_DIR, "pull"])
        except subprocess.CalledProcessError:
            print("   ⚠️ git pull a échoué, on repart d'un clone propre.")
            shutil.rmtree(PROJECT_DIR)
            run(["git", "clone", GITHUB_REPO_URL, PROJECT_DIR])
    else:
        try:
            run(["git", "clone", GITHUB_REPO_URL, PROJECT_DIR])
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ git n'est pas disponible ou le clone a échoué.")
            print("   Installez Git depuis https://git-scm.com/downloads puis relancez.")
            sys.exit(1)


def setup_dataset():
    """Télécharge un extrait Wikipédia FR si le dataset local est absent.
    Exécuté via un sous-processus avec PYTHON_CMD (et non importé
    directement), car dans un .exe figé l'interpréteur embarqué n'a pas
    accès aux paquets installés dans le Python système du volontaire."""
    data_path = os.path.join(PROJECT_DIR, "data", "input.txt")
    if os.path.exists(data_path) and os.path.getsize(data_path) > 0:
        print(f"\n✅ Dataset local déjà présent ({data_path})")
        return

    print(f"\n⏳ Téléchargement d'un extrait Wikipédia FR ({WIKI_ARTICLES_COUNT} articles)...")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    download_script = (
        "from datasets import load_dataset\n"
        "ds = load_dataset('wikimedia/wikipedia', '20231101.fr', split='train', streaming=True)\n"
        "written = 0\n"
        f"with open(r'{data_path}', 'w', encoding='utf-8') as f:\n"
        "    for i, article in enumerate(ds):\n"
        "        f.write('### Source: ' + article['url'] + chr(10))\n"
        "        f.write(article['text'] + chr(10) + chr(10))\n"
        "        written += 1\n"
        "        if written % 100 == 0:\n"
        "            f.flush()\n"
        "            print('   ...', written, 'articles téléchargés')\n"
        f"        if i >= {WIKI_ARTICLES_COUNT}:\n"
        "            break\n"
        "print('DONE:', written)\n"
    )
    run([PYTHON_CMD, "-c", download_script])
    print(f"✅ Dataset prêt dans {data_path}")


def launch_training():
    print("\n🚀 Lancement de l'entraînement décentralisé...")
    print(f"   Connexion au pair initial : {INITIAL_PEER_ADDR}\n")
    os.chdir(PROJECT_DIR)
    run([
        PYTHON_CMD, "train.py",
        "--initial-peers", INITIAL_PEER_ADDR,
    ])


def main():
    global PYTHON_CMD
    print("=" * 70)
    print("🤝 Installation automatique — Entraînement GPT français décentralisé")
    print("=" * 70)

    check_python()
    check_git()
    PYTHON_CMD = get_python_command()
    print(f"🐍 Python système utilisé : {PYTHON_CMD}")
    install_dependencies()
    setup_project_code()
    setup_dataset()
    launch_training()


if __name__ == "__main__":
    main()