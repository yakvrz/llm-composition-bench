#!/usr/bin/env python3
"""
Wrapper for the explicit (candidate-based) evaluator.
Delegates to the repository root `evaluate.py`.
"""
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import evaluate as _eval
    _eval.main()


if __name__ == "__main__":
    main()


