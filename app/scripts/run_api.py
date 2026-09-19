"""
Start the FastAPI server.

Usage:
    python -m app.scripts.run_api
    python -m app.scripts.run_api --port 8000 --reload
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from app.config import settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the RAG API server.")
    parser.add_argument("--host", default=settings.API_HOST)
    parser.add_argument("--port", type=int, default=settings.API_PORT)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  RAG System -- API Server")
    print("=" * 60)
    print(f"  Host:       {args.host}")
    print(f"  Port:       {args.port}")
    print(f"  Reload:     {args.reload}")
    print(f"  Workers:    {args.workers}")
    print(f"  Log level:  {args.log_level}")
    print()
    print(f"  UI:         http://{args.host}:{args.port}")
    print(f"  API docs:   http://{args.host}:{args.port}/docs")
    print("=" * 60)
    print()

    try:
        uvicorn.run(
            "app.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level,
        )
        return 0
    except KeyboardInterrupt:
        print("\nServer stopped.")
        return 0
    except Exception as exc:
        print(f"Server error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
