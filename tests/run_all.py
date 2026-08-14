"""Run all SPA-RL tests (pure Python, no GPU / model downloads needed)."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "test_grounding.py",
    "test_reward.py",
    "test_annotate.py",
    "test_rl_data_org.py",
    "test_metrics.py",
    "test_chatml_mask.py",
]


def main():
    failures = 0
    for name in TESTS:
        print(f"\n=== {name} ===")
        result = subprocess.run(
            [sys.executable, str(ROOT / "tests" / name)],
            cwd=ROOT,
        )
        if result.returncode != 0:
            failures += 1
    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} TEST FILE(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
