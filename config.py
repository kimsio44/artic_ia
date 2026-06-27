from dataclasses import dataclass

@dataclass
class GPTConfig:
    # Architecture
    block_size: int = 256       # longueur max du contexte
    vocab_size: int = 50257     # vocabulaire GPT-2 (tiktoken)
    n_layer: int = 12           # nombre de blocs transformer
    n_head: int = 12            # têtes d'attention
    n_embd: int = 768           # dimension des embeddings
    dropout: float = 0.1
    bias: bool = True

@dataclass
class TrainConfig:
    # Entraînement
    batch_size: int = 4
    max_steps: int = 5000
    eval_interval: int = 200
    eval_steps: int = 50
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Checkpointing
    out_dir: str = "checkpoints"
    save_interval: int = 500

    # Dataset
    data_path: str = "data/input.txt"
    train_split: float = 0.9