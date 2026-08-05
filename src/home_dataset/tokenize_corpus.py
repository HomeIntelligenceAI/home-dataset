"""Tokenise a JSONL corpus into the flat uint16 stream the trainer reads.

Encoding is pure-Python BPE at ~268k chars/s on one core, so a 1.3B-character
corpus is about 1.4 hours single-threaded and ~10 minutes across 12. The work
is embarrassingly parallel — documents are independent — so it is farmed out
with a process pool.

Run as:
    python -m home_dataset.tokenize_corpus \
        --input corpus.jsonl --vocab vocab.json --out-dir tokens/
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path

import numpy as np

__all__ = ["tokenize_corpus"]

_TOKENIZER = None
_EOS: int | None = None


def _init_worker(vocab_path: str) -> None:
    """Load the tokenizer once per worker rather than once per document.

    The tokenizer holds a 31,762-entry merge-rank table; rebuilding it per
    task would cost more than the encoding.
    """
    global _TOKENIZER, _EOS
    from home_tokenizer import HomeTokenizer

    _TOKENIZER = HomeTokenizer.from_trained(vocab_path, add_special_tokens=False)
    _EOS = _TOKENIZER.vocab.eos_id


def _encode_chunk(texts: list[str]) -> list[int]:
    assert _TOKENIZER is not None
    out: list[int] = []
    for text in texts:
        out.extend(_TOKENIZER.encode(text))
        if _EOS is not None:
            # Documents are separated by EOS so the model learns where one
            # stops, rather than learning to run them together.
            out.append(_EOS)
    return out


def _chunks(
    paths: list[Path], size: int, limit: int | None
) -> Iterator[list[str]]:
    """Stream documents from one or more JSONL files as fixed-size batches.

    Multiple inputs are concatenated in the order given, so the validation tail
    comes from the last source listed. Put the most representative source last.
    """
    batch: list[str] = []
    seen = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if limit is not None and seen >= limit:
                    break
                seen += 1
                text = json.loads(line).get("text")
                if text:
                    batch.append(text)
                if len(batch) >= size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def tokenize_corpus(
    input_paths: list[Path],
    vocab_path: Path,
    out_dir: Path,
    workers: int = 8,
    chunk_size: int = 500,
    val_fraction: float = 0.002,
    limit: int | None = None,
) -> dict[str, object]:
    """Tokenise to disk incrementally.

    Everything is streamed. An earlier version accumulated the whole corpus in
    a Python list before writing, which is fine at 280M tokens and fatal at 1B:
    a Python int is ~28 bytes, so a billion of them is ~28 GB of RAM. It died
    with MemoryError 55 minutes into a run.
    """
    from home_training.data import TOKEN_DTYPE, DataStats

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out_dir / "train.bin", out_dir / "val.bin"
    started = time.perf_counter()

    total = 0
    first_block: np.ndarray | None = None

    with (
        Pool(workers, initializer=_init_worker, initargs=(str(vocab_path),)) as pool,
        train_path.open("wb") as sink,
    ):
        for done, part in enumerate(
            # imap keeps ordering, which matters: the val split is taken from
            # the tail, and a shuffled stream would put training text in it.
            pool.imap(_encode_chunk, _chunks(input_paths, chunk_size, limit)),
            start=1,
        ):
            block = np.asarray(part, dtype=TOKEN_DTYPE)
            block.tofile(sink)
            total += block.size
            if first_block is None:
                first_block = block
            if done % 40 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  {done * chunk_size:>9,} docs  {total / 1e6:>7.1f}M tokens  "
                    f"{elapsed:>5.0f}s  {total / elapsed / 1e3:>6.0f}k tok/s",
                    flush=True,
                )

    # Split by moving the tail into val.bin, then truncating train.bin --
    # no second pass over the data and no second copy on disk.
    val_count = max(1, int(total * val_fraction))
    split_at = total - val_count
    stream = np.memmap(train_path, dtype=TOKEN_DTYPE, mode="r")
    np.asarray(stream[split_at:]).tofile(val_path)
    del stream
    os.truncate(train_path, split_at * np.dtype(TOKEN_DTYPE).itemsize)

    sample = np.memmap(train_path, dtype=TOKEN_DTYPE, mode="r")[:5_000_000]
    unique = int(np.unique(np.asarray(sample)).size)
    del sample

    return {
        "train": DataStats(split_at, train_path.stat().st_size),
        "val": DataStats(val_count, val_path.stat().st_size),
        "elapsed_s": time.perf_counter() - started,
        "distinct_tokens_in_first_5M": unique,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenise a JSONL corpus")
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="One or more JSONL files, concatenated in order")
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.002)
    args = parser.parse_args()

    # A source that failed to download should not stop the ones that
    # succeeded. Listing all five and having the command die on the one the
    # network dropped is exactly the wrong failure mode for a 2-hour job.
    present = [p for p in args.input if p.exists()]
    for missing in [p for p in args.input if not p.exists()]:
        print(f"  skipping {missing.name} -- not found", flush=True)
    if not present:
        raise SystemExit("none of the given input files exist")

    total_gb = sum(p.stat().st_size for p in present) / 1e9
    print(f"tokenising {len(present)} file(s), {total_gb:.1f} GB", flush=True)
    args.input = present
    result = tokenize_corpus(
        args.input,
        args.vocab,
        args.out_dir,
        workers=args.workers,
        val_fraction=args.val_fraction,
        limit=args.limit,
    )
    print(f"\ntrain {result['train']}")
    print(f"val   {result['val']}")
    print(f"distinct token ids seen (first 5M): {result['distinct_tokens_in_first_5M']:,}")
    print(f"elapsed {result['elapsed_s']:.0f}s")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
