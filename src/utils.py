"""Utility functions and helpers."""
import hashlib
import logging
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

import config


def setup_logging(name: str = "courseware_qa") -> logging.Logger:
    """Configure and return a logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    log_file = config.LOGS_DIR / f"{name}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def format_citation(file_name: str, page_num: int) -> str:
    """Format a citation string."""
    return f"[来源：{file_name}，第 {page_num} 页]"


def normalize_latex(text: str) -> str:
    r"""Convert LaTeX delimiters to Streamlit-compatible format.

    Streamlit's markdown renderer (KaTeX) supports $...$ for inline math
    and $$...$$ for display math, but NOT \(...\) or \[...\] which
    DeepSeek and other LLMs often output.

    Transformation rules:
        \[ expr \]  ->  $$expr$$   (display / block math)
        \( expr \)  ->  $expr$     (inline math)
    """
    # Display math: \[ ... \]  →  $$ ... $$
    text = re.sub(r'\\\[', '$$', text)
    text = re.sub(r'\\\]', '$$', text)
    # Inline math: \( ... \)  →  $ ... $
    text = re.sub(r'\\\(', '$', text)
    text = re.sub(r'\\\)', '$', text)
    return text
