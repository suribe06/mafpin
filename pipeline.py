"""
Backward-compatible entry point for the MAFPIN pipeline.

All logic lives in the ``pipeline/`` package.  Run directly::

    python pipeline.py --all
    python pipeline.py --steps cascade inference

Or use the package form::

    python -m pipeline --all
"""

from pipeline import main

if __name__ == "__main__":
    main()
