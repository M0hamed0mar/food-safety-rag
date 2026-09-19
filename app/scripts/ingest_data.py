"""
Bulk-ingest documents from a folder into the RAG system.

Usage:
    python -m app.scripts.ingest_data
    python -m app.scripts.ingest_data --input-dir data/uploaded_docs
    python -m app.scripts.ingest_data --input-dir path/to/docs --recursive
    python -m app.scripts.ingest_data --file path/to/file.pdf

Behavior:
    - Resets any existing vector index (fresh start)
    - Scans for supported file types
    - Ingests each file through the full pipeline
    - Prints a summary at the end
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from app.config import settings
from app.config.constants import SUPPORTED_EXTENSIONS
from app.monitoring import get_logger, setup_logging
from app.pipeline import get_pipeline, reset_pipeline


logger = get_logger("app.scripts.ingest_data")


def find_files(input_dir: Path, recursive: bool = True) -> list[Path]:
    """Return all supported files under input_dir."""
    if not input_dir.exists():
        return []

    pattern = "**/*" if recursive else "*"
    files: list[Path] = []
    for p in input_dir.glob(pattern):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)
    return sorted(files)


def print_header() -> None:
    print()
    print("=" * 68)
    print("  RAG System -- Document Ingestion")
    print("=" * 68)
    print()


def print_summary(
    total_files: int,
    successful: int,
    failed: int,
    total_chunks: int,
    elapsed_s: float,
) -> None:
    print()
    print("=" * 68)
    print("  Ingestion Summary")
    print("=" * 68)
    print(f"  Total files:       {total_files}")
    print(f"  Successful:        {successful}")
    print(f"  Failed:            {failed}")
    print(f"  Total chunks:      {total_chunks}")
    print(f"  Elapsed:           {elapsed_s:.2f}s")
    print("=" * 68)
    print()


def ingest_file(pipeline, file_path: Path, index: int, total: int) -> tuple[bool, int]:
    """Ingest a single file. Returns (success, chunk_count)."""
    prefix = f"[{index}/{total}]"
    name = file_path.name

    try:
        size_kb = file_path.stat().st_size / 1024
    except OSError:
        size_kb = 0.0

    print(f"{prefix} {name}  ({size_kb:.1f} KB)")

    try:
        result = pipeline.ingest_document(file_path)
        chunks = result.chunk_count or 0
        duration = result.get_total_duration_ms() or 0.0
        print(f"        -> status: {result.status}, chunks: {chunks}, "
              f"duration: {duration:.0f} ms")
        return True, chunks
    except Exception as exc:
        print(f"        -> FAILED: {type(exc).__name__}: {exc}")
        logger.log_error(
            event=__import__("app.config.constants", fromlist=["LogEvent"]).LogEvent.ERROR,
            message=f"Ingestion failed for {name}: {exc}",
            exception=exc,
        )
        return False, 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bulk-ingest documents into the RAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=settings.DATA_DIR / "uploaded_docs",
        help="Directory to scan for documents (default: data/uploaded_docs)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Ingest a single file instead of scanning a directory",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse into subdirectories",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not reset the existing index before ingesting",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Logging
    setup_logging(log_level=args.log_level)

    print_header()

    # ---- Resolve files ----
    if args.file is not None:
        if not args.file.exists():
            print(f"Error: file does not exist: {args.file}")
            return 1
        files = [args.file]
    else:
        if not args.input_dir.exists():
            print(f"Error: input dir does not exist: {args.input_dir}")
            return 1
        files = find_files(args.input_dir, recursive=not args.no_recursive)

    if not files:
        print(f"No supported documents found.")
        print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return 0

    print(f"Found {len(files)} document(s) to ingest.")
    print()

    # ---- Fresh start (optional) ----
    if not args.keep_existing:
        print("Resetting pipeline (fresh start)...")
        reset_pipeline()

        # Wipe vector store data (FAISS + metadata)
        from app.vector_store import FAISSStore, MetadataStore

        faiss_path = settings.FAISS_INDEX_DIR
        if faiss_path.exists():
            for f in faiss_path.glob("*.faiss"):
                f.unlink()
            for f in faiss_path.glob("*.meta"):
                f.unlink()
            for f in faiss_path.glob("*.db"):
                f.unlink()
            print("  Removed previous vector index files.")
        print()

    # ---- Get pipeline ----
    print("Initializing pipeline...")
    t0 = time.perf_counter()
    try:
        pipeline = get_pipeline()
    except Exception as exc:
        print(f"Error: pipeline initialization failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    init_time = time.perf_counter() - t0
    print(f"Pipeline ready in {init_time:.2f}s.")
    print()

    # ---- Ingest each file ----
    successful = 0
    failed = 0
    total_chunks = 0

    for i, file_path in enumerate(files, start=1):
        ok, chunks = ingest_file(pipeline, file_path, i, len(files))
        if ok:
            successful += 1
            total_chunks += chunks
        else:
            failed += 1

    elapsed = time.perf_counter() - t0
    print_summary(len(files), successful, failed, total_chunks, elapsed)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
