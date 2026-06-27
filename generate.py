import os
import pickle
import torch
import tiktoken
import argparse
from torch.serialization import safe_globals
from config import GPTConfig
from model import GPT
from data import get_device


def load_model(checkpoint_path: str, device):
    """Charge un modèle depuis un checkpoint."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}\n"
            "Entraîne d'abord le modèle avec : python train.py"
        )

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except (pickle.UnpicklingError, RuntimeError):
        with safe_globals([GPTConfig]):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt["gpt_config"]
    if isinstance(config, dict):
        config = GPTConfig(**config)

    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    step = ckpt.get("step", "?")
    val_loss = ckpt.get("val_loss", "?")
    print(f"✅ Modèle chargé — step {step} | val loss {val_loss}")
    return model


@torch.no_grad()
def generate_text(
    model,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    device=None,
):
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    print(f"\n📝 Prompt : {prompt!r}")
    print(f"⚙️  Température : {temperature} | Top-k : {top_k} | Tokens max : {max_new_tokens}")
    print("\n" + "─" * 60)

    output = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    generated = enc.decode(output[0].tolist())
    print(generated)
    print("─" * 60)
    return generated


def interactive_mode(model, device):
    """Mode interactif : entre un prompt, reçoit une réponse."""
    enc = tiktoken.get_encoding("gpt2")
    print("\n🤖 Mode interactif (tape 'quit' pour quitter)\n")

    while True:
        prompt = input("Prompt > ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break
        if not prompt:
            continue

        generate_text(model, prompt, device=device)


def main():
    parser = argparse.ArgumentParser(description="Génération de texte avec GPT-2 custom")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Chemin vers le checkpoint")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Texte de départ pour la génération")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Nombre de tokens à générer")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Température (0.1 = déterministe, 1.5 = créatif)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling (0 = désactivé)")
    parser.add_argument("--interactive", action="store_true",
                        help="Mode interactif")
    args = parser.parse_args()

    device = get_device()
    model  = load_model(args.checkpoint, device)

    if args.interactive:
        interactive_mode(model, device)
    elif args.prompt:
        generate_text(
            model,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            device=device,
        )
    else:
        # Prompt par défaut
        generate_text(model, prompt="Il était une fois", device=device)


if __name__ == "__main__":
    main()