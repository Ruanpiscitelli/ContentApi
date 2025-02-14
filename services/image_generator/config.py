import torch
import os

MODEL_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "compile": True,
    "use_native_attention": True,
    "use_opt_einsum": True,
    "batch_size": int(os.getenv("BATCH_SIZE", "4")),
    "num_workers": min(4, os.cpu_count() or 1),
} 