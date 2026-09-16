"""Serve the Cognitive Load Predictor V2 HTTP API (FR-9).

Usage:
    python serve_api.py [--host 127.0.0.1] [--port 8000] [--reload]

Then try:
    curl http://127.0.0.1:8000/api/v1/health
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the V2 HTTP API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev).")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "cognitive_load.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
