"""Logging utilities: tee-stream and log-file helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TextIO

from config import DatasetPaths as _ConfigDatasetPaths


class _TeeStream:
    """Write to both a primary stream and a log file simultaneously."""

    def __init__(self, primary: TextIO, log_file: TextIO) -> None:
        self.primary = primary
        self.log_file = log_file
        self.encoding = getattr(primary, "encoding", "utf-8")

    def write(self, data: str) -> int:
        self.primary.write(data)
        self.log_file.write(data)
        self.flush()
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self.primary, name)


def _default_log_path(dataset: str) -> Path:
    return _ConfigDatasetPaths(dataset).BASE / "pipeline.log"


def _open_pipeline_log(args: argparse.Namespace) -> TextIO | None:
    if args.no_log:
        return None
    log_path = Path(args.log_file) if args.log_file else _default_log_path(args.dataset)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    print(f"Pipeline log: {log_path}", flush=True)
    return log_file
