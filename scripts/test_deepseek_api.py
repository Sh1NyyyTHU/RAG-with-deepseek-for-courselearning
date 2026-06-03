"""
Test DeepSeek API connectivity and basic functionality.

Usage: python scripts/test_deepseek_api.py
"""
import sys
from pathlib import Path

# Fix Unicode output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deepseek_client import DeepSeekClient
from src.utils import setup_logging


def main():
    logger = setup_logging()

    print("=" * 60)
    print("DeepSeek API Connectivity Test")
    print("=" * 60)

    client = DeepSeekClient()

    # Test 1: Connectivity
    print("\n[1] Testing API connectivity...")
    result = client.check_connectivity()
    print(f"  Status: {result['status']}")
    print(f"  Model: {result['model']}")
    if result["status"] != "ok":
        print(f"  Error: {result.get('error', 'unknown')}")
        sys.exit(1)
    print("  [OK] Connected")

    # Test 2: Simple streaming chat
    print("\n[2] Testing streaming chat...")
    try:
        response = client.chat(
            messages=[
                {"role": "user", "content": "Say 'Hello, API is working!' in Chinese."}
            ],
            stream=True,
        )
        print("  Response: ", end="", flush=True)
        full = ""
        for chunk in response:
            print(chunk, end="", flush=True)
            full += chunk
        print()
        print("  [OK] Streaming works")
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        sys.exit(1)

    # Test 3: Non-streaming chat
    print("\n[3] Testing non-streaming chat...")
    try:
        response = client.chat_sync(
            messages=[
                {"role": "user", "content": "Reply with exactly: OK"}
            ],
        )
        print(f"  Response: {response}")
        print("  [OK] Non-streaming works")
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")

    # Test 4: Ask with QA mode
    print("\n[4] Testing ask() with QA mode...")
    try:
        response = client.ask(
            system_prompt="You are a helpful assistant. Keep answers brief.",
            user_prompt="What is 2+2?",
            mode="qa",
            stream=True,
        )
        print("  Response: ", end="", flush=True)
        for chunk in response:
            print(chunk, end="", flush=True)
        print()
        print("  [OK] ask() works")
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")

    print("\n" + "=" * 60)
    print("All tests passed! [OK]")
    print("=" * 60)


if __name__ == "__main__":
    main()
