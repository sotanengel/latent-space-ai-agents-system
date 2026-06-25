"""Global pytest configuration.

Keeps Hugging Face / Torch noise out of test output and gives the integration
tests a small CPU-only model to work with.
"""

from __future__ import annotations

import os

# Force CPU + offline-friendly defaults before torch / transformers import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

TINY_MODEL_ID = "sshleifer/tiny-gpt2"
