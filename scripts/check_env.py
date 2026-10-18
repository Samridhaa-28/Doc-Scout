"""Import every project library, print versions and the torch device, then 'all good'. Logic: docscout.env_check.

Usage: .venv/Scripts/python -m scripts.check_env
"""
from __future__ import annotations

import sys

from docscout.env_check import check_env

if __name__ == "__main__":
    sys.exit(check_env())
