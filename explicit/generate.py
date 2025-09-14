#!/usr/bin/env python3
"""
Wrapper for the explicit (candidate-based) generator.
Delegates to the repository root `generate_data.py`.
"""
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import generate_data as _gen
    _gen.main()


if __name__ == "__main__":
    main()


