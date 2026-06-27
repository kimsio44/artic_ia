"""
train_hivemind.py — Variante décentralisée de train.py utilisant Hivemind.

Architecture :
- hivemind.DHT : réseau pair-à-pair (table de hachage distribuée) qui permet
  aux pairs de se découvrir sans serveur central.
- hivemind.Optimizer : enveloppe votre optimiseur PyTorch habituel et
  synchronise les gradients avec les autres pairs en arrière-plan, de façon
  tolérante aux pannes (un pair qui plante ou se déconnecte n'arrête pas
  les autres).

⚠️ IMPORTANT — à tester d'abord EN LOCAL (2 terminaux sur la même machine),
avant d'ouvrir à de vrais volontaires sur internet. Voir les instructions
en bas de ce fichier.

Installation requise :
    pip install hivemind --break-system-packages
"""

import os
import argparse
import multiprocessing
import torch
import hivemind
from config import GPTConfig, TrainConfig
from model import GPT
from data import TextDataset, create_sample_data

# Sur macOS, Python 3.8+ utilise 'spawn' par défaut pour les nouveaux
# processus, ce qui nécessite de sérialiser (pickle) tout l'état du
# processus parent. Hivemind crée en interne des objets non-picklables
# (verrous de thread), ce qui plante avec 'spawn'. On force donc 'fork'
# (comportement historique, utilisé par défaut sur Linux), qui évite ce
# problème de sérialisation. Doit être fait avant toute autre opération
# de multiprocessing dans le script.
if multiprocessing.get_start_method(allow_none=True) != "fork":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass  # déjà fixé ailleurs (ex: import répété), on ignore


def evaluate(model, dataset, train_cfg, device):
    """Identique à train.py — calcule la loss sur train et val sans gradient."""
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        total_loss = 0.0
        for _ in range(train_cfg.eval_steps):
            x, y = dataset.get_batch(split, model.config.block_size, train_cfg.batch_size, device)
            with torch.no_grad():
                _, loss = model(x, y)
            total_loss += loss.item()
        losses[split] = total_loss / train_cfg.eval_steps
    model.train()
    return losses


def train_decentralized(initial_peers=None, host_maddrs=None):
    # ── Configs (identiques à train.py) ──────────────────────
    gpt_cfg = GPTConfig()
    train_cfg = TrainConfig()

    # ⚠️ Hivemind utilise du multiprocessing en interne (state averager)
    # qui requiert torch.Storage._share_filename_cpu_, NON supporté par
    # le backend MPS (Apple Silicon). On force donc CPU ici, même si
    # get_device() détecterait MPS — c'est une limitation connue de
    # l'intégration Hivemind/MPS, pas un bug de configuration.
    device = torch.device("cpu")
    print("⚠️  Mode décentralisé : CPU forcé (MPS incompatible avec le "
          "multiprocessing interne de Hivemind sur Apple Silicon)")

    os.makedirs(train_cfg.out_dir, exist_ok=True)

    # ── DHT : le réseau décentralisé ──────────────────────────
    # Si initial_peers est vide, ce pair DÉMARRE un nouveau réseau (premier
    # pair). Les pairs suivants doivent fournir l'adresse de ce premier
    # pair pour le rejoindre.
    dht_kwargs = {"start": True}
    if initial_peers:
        dht_kwargs["initial_peers"] = initial_peers
    # host_maddrs par défaut : port FIXE (pas aléatoire), nécessaire pour
    # configurer le port forwarding du routeur de façon stable.
    # Port 14000 choisi car certains routeurs (ex: Freebox) limitent les
    # redirections WAN à la plage 1-16383 — adaptez si besoin, en gardant
    # la même valeur ici, dans le Dockerfile (EXPOSE) et dans votre règle
    # de port forwarding.
    dht_kwargs["host_maddrs"] = host_maddrs or [
        "/ip4/0.0.0.0/tcp/14000",
        "/ip4/0.0.0.0/udp/14000/quic",
    ]

    dht = hivemind.DHT(**dht_kwargs)

    print("\n" + "=" * 70)
    print("🌐 DHT démarré. Pour qu'un AUTRE pair rejoigne CE réseau,")
    print("   il doit lancer son script avec --initial-peers suivi de :")
    for addr in dht.get_visible_maddrs():
        print(f"   {addr}")
    print("=" * 70 + "\n")

    # ── Dataset (identique à train.py) ────────────────────────
    create_sample_data(train_cfg.data_path)
    dataset = TextDataset(train_cfg.data_path, train_cfg.train_split)

    # ── Modèle (identique à train.py) ─────────────────────────
    model = GPT(gpt_cfg).to(device)

    # ── Optimiseur de base (identique à train.py) ─────────────
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]

    base_optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.learning_rate,
    )

    # ── hivemind.Optimizer : enveloppe l'optimiseur pour synchroniser
    #    les gradients avec les autres pairs en arrière-plan ───────────
    # target_batch_size : taille de batch GLOBALE visée, cumulée sur tous
    # les pairs connectés (pas juste le batch local). Avec plusieurs pairs,
    # chacun contribue une fraction de ce total à chaque "epoch" logique.
    optimizer = hivemind.Optimizer(
        dht=dht,
        run_id="gpt_custom_fr_decentralized",  # doit être IDENTIQUE sur tous les pairs
        optimizer=base_optimizer,
        target_batch_size=train_cfg.batch_size * 8,  # à ajuster selon le nb de pairs attendu
        batch_size_per_step=train_cfg.batch_size,
        matchmaking_time=15.0,   # secondes d'attente pour regrouper les pairs avant une étape
        averaging_timeout=30.0,  # délai max pour la moyenne des gradients entre pairs
        verbose=True,
    )

    print(f"\n🚀 Début de l'entraînement décentralisé sur {device}")
    print(f"   Steps : {train_cfg.max_steps} | Batch local : {train_cfg.batch_size} | LR : {train_cfg.learning_rate}\n")

    best_val_loss = float("inf")
    step = 0

    while step < train_cfg.max_steps:
        if step % train_cfg.eval_interval == 0:
            losses = evaluate(model, dataset, train_cfg, device)
            print(f"Step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} "
                  f"| pairs visibles : {len(dht.get_visible_maddrs())}")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                ckpt_path = os.path.join(train_cfg.out_dir, "best_model.pt")
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "val_loss": best_val_loss,
                    "gpt_config": gpt_cfg.__dict__,
                }, ckpt_path)
                print(f"         Meilleur modèle sauvegardé (val loss {best_val_loss:.4f})")

        x, y = dataset.get_batch("train", gpt_cfg.block_size, train_cfg.batch_size, device)
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()  # synchronise avec les autres pairs en arrière-plan

        # local_step est géré par hivemind.Optimizer ; on utilise son
        # compteur global plutôt qu'un simple compteur local, car le
        # rythme réel dépend de la disponibilité des autres pairs.
        step = optimizer.local_epoch if hasattr(optimizer, "local_epoch") else step + 1

        if step > 0 and step % train_cfg.save_interval == 0:
            ckpt_path = os.path.join(train_cfg.out_dir, f"ckpt_step{step}.pt")
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "gpt_config": gpt_cfg.__dict__,
            }, ckpt_path)

    print("\n✅ Entraînement décentralisé terminé !")
    print(f"   Meilleure val loss : {best_val_loss:.4f}")
    print(f"   Checkpoints dans  : {train_cfg.out_dir}/")

    optimizer.shutdown()
    dht.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement décentralisé GPT custom avec Hivemind")
    parser.add_argument(
        "--initial-peers", type=str, nargs="*", default=None,
        help="Adresses des pairs existants à rejoindre (laisser vide pour démarrer un nouveau réseau)"
    )
    parser.add_argument(
        "--host-maddrs", type=str, nargs="*", default=None,
        help="Adresses d'écoute locales, ex: /ip4/0.0.0.0/tcp/0 (par défaut : port aléatoire)"
    )
    args = parser.parse_args()
    train_decentralized(initial_peers=args.initial_peers, host_maddrs=args.host_maddrs)