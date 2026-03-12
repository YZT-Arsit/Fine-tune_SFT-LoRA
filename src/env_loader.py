from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    dotenv_path = project_root / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return dotenv_path
