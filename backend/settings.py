"""Loads config.yaml and secrets from .env."""
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


@lru_cache
def config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def secret(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def secrets(name: str) -> list[str]:
    """NAME, then backups NAME_2 ... NAME_5 (e.g. teammates' keys), in that order, skipping unset ones."""
    return [v for v in [secret(name)] + [secret(f"{name}_{i}") for i in range(2, 6)] if v]
