"""
Central configuration for the Courseware QA system.
All settings are loaded from environment variables with sensible defaults.
"""
import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv(Path(__file__).parent / ".env", override=False)

# ── Project paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma_db"
EXPORTS_DIR = DATA_DIR / "exports"
LOGS_DIR = DATA_DIR / "logs"

# Ensure directories exist
for d in [DATA_DIR, PDF_DIR, CHROMA_DIR, EXPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── DeepSeek API ───────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_TIMEOUT = 120  # seconds
DEEPSEEK_MAX_RETRIES = 2
DEEPSEEK_STREAMING = True

# ── Embedding ──────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_MODEL_DIR = os.getenv("EMBEDDING_MODEL_DIR", None)

# CUDA detection
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    EMBEDDING_DEVICE = "cuda"
    EMBEDDING_USE_FP16 = True
else:
    EMBEDDING_DEVICE = "cpu"
    EMBEDDING_USE_FP16 = False

# ── Text splitting ─────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# ── Retrieval ──────────────────────────────────────────────────
DEFAULT_TOP_K = 8
SIMILARITY_THRESHOLD = 0.0  # can be adjusted to filter low-relevance results

# ── Logging ─────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
