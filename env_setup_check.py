"""
Pre-flight check script. Run before launching the app.
Verifies all dependencies and environment are correct.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Fix Unicode output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 60)
print("Courseware QA System — Pre-flight Check")
print("=" * 60)

errors = []
warnings = []

# 1. Python version
print("\n[1] Python version:", sys.version)
if sys.version_info < (3, 10):
    errors.append("Python 3.10+ required")

# 2. Core imports
print("\n[2] Core dependencies:")
deps = {
    "streamlit": "streamlit",
    "chromadb": "chromadb",
    "pymupdf (fitz)": "fitz",
    "openai": "openai",
    "FlagEmbedding": "FlagEmbedding",
    "torch": "torch",
    "dotenv": "dotenv",
    "numpy": "numpy",
}
for name, module in deps.items():
    try:
        __import__(module)
        print(f"  [OK] {name}")
    except ImportError:
        print(f"  [FAIL] {name}")
        errors.append(f"{name} not installed")

# 3. CUDA
print("\n[3] CUDA status:")
try:
    import torch
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        warnings.append("CUDA not available — using CPU")
except Exception as e:
    warnings.append(f"CUDA check failed: {e}")

# 4. API key
print("\n[4] DeepSeek API key:")
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=False)
api_key = os.getenv("DEEPSEEK_API_KEY", "")
if api_key and len(api_key) > 10:
    print(f"  [OK] API key found (starts with: {api_key[:8]}...)")
else:
    warnings.append("DEEPSEEK_API_KEY not set — API features will fail")
    print("  [WARN] DEEPSEEK_API_KEY not set")
    print("  Create a .env file with: DEEPSEEK_API_KEY=your_key_here")

# 5. Data directories
print("\n[5] Data directories:")
for d in ["data", "data/pdfs", "data/chroma_db", "data/exports", "data/logs"]:
    path = Path(__file__).parent / d
    path.mkdir(parents=True, exist_ok=True)
    print(f"  [OK] {d}/")

# 6. Summary
print("\n" + "=" * 60)
if errors:
    print(f"[FAIL] {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
else:
    print("[OK] No critical errors")

if warnings:
    print(f"[WARN] {len(warnings)} warning(s):")
    for w in warnings:
        print(f"  - {w}")

print("=" * 60)

if errors:
    sys.exit(1)
else:
    print("\nSystem is ready to start (run: streamlit run app.py)")
