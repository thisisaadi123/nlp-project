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
