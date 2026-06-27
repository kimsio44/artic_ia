import os
import torch
import tiktoken
import numpy as np
from config import TrainConfig


def get_device():
    """Détecte le meilleur device disponible (MPS pour M1, CUDA, ou CPU)."""
    if torch.backends.mps.is_available():
        print(" Apple MPS (Metal) détecté — GPU M1 activé")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(" CUDA détecté")
        return torch.device("cuda")
    else:
        print("  CPU uniquement")
        return torch.device("cpu")


class TextDataset:
    """
    Charge un fichier texte, le tokenise avec tiktoken (GPT-2)
    et le découpe en train / validation.
    """

    def __init__(self, data_path: str, train_split: float = 0.9):
        self.enc = tiktoken.get_encoding("gpt2")

        assert os.path.exists(data_path), \
            f"Fichier introuvable : {data_path}\n" \
            f"Place ton texte dans data/input.txt"

        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"Dataset chargé — {len(text):,} caractères")

        tokens = self.enc.encode(text)
        data = torch.tensor(tokens, dtype=torch.long)
        print(f"Tokens totaux : {len(data):,}")

        # Split train / val
        n = int(len(data) * train_split)
        self.train_data = data[:n]
        self.val_data   = data[n:]
        print(f"Train : {len(self.train_data):,} tokens | Val : {len(self.val_data):,} tokens")

    def get_batch(self, split: str, block_size: int, batch_size: int, device):
        """Retourne un batch (x, y) aléatoire depuis train ou val."""
        data = self.train_data if split == "train" else self.val_data

        if len(data) <= block_size:
            raise ValueError(
                f"Dataset trop petit ({len(data)} tokens) pour block_size={block_size}. "
                f"Utilise un texte plus long ou réduis block_size."
            )

        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)


def create_sample_data(path: str = "data/input.txt"):
    """Crée un fichier exemple si aucun dataset n'est fourni."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(path):
        sample = (
            "Il était une fois, dans un royaume lointain, un jeune développeur "
            "qui voulait entraîner son propre modèle de langage. "
            "Il avait un MacBook M1 et une grande ambition. "
            "Chaque jour, il écrivait du code et apprenait de nouvelles choses. "
            "Les transformers n'avaient plus de secrets pour lui. "
        ) * 500  # répéter pour avoir assez de tokens
        with open(path, "w", encoding="utf-8") as f:
            f.write(sample)
        print(f"✅ Fichier exemple créé : {path}")
    else:
        print(f"✅ Dataset existant trouvé : {path}")