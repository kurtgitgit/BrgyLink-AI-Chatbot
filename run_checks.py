"""One fail-fast command for the supported local chatbot regression suites."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CHECKS = (
    "test_resident_readiness.py", "test_chat_api.py", "test_security.py",
    "test_freshness.py", "test_architecture.py", "test_integration_training.py",
    "test_holdout.py", "test_multilingual_all.py",
)


def main() -> int:
    for script in CHECKS:
        print(f"\nChecking {script}", flush=True)
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / script)], cwd=ROOT)
        if result.returncode:
            print(f"FAILED: {script}; do not publish or claim readiness.", flush=True)
            return result.returncode
    print("\nAll 8 local suites passed. This is not proof of production/device behavior.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
