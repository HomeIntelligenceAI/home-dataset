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


def _chunks(path: Path, size: int, limit: int | None) -> Iterator[list[str]]:
    batch: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if limit is not None and i >= limit:
                break
            text = json.loads(line).get("text")
            if text:
                batch.append(text)
            if len(batch) >= size:
                yield batch
                batch = []
    if batch:
        yield batch


def tokenize_corpus(
    input_path: Path,
    vocab_path: Path,
    out_dir: Path,
    workers: int = 8,
    chunk_size: int = 500,
    val_fraction: float = 0.002,
    limit: int | None = None,
) -> dict[str, object]:
    from home_training.data import write_tokens

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tokens: list[int] = []

    with Pool(workers, initializer=_init_worker, initargs=(str(vocab_path),)) as pool:
        for done, part in enumerate(
            # imap keeps ordering, which matters: the val split is taken from
            # the tail, and a shuffled stream would put training text in it.
            pool.imap(_encode_chunk, _chunks(input_path, chunk_size, limit)),
            start=1,
        ):
            tokens.extend(part)
            if done % 40 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  {done * chunk_size:>9,} docs  {len(tokens) / 1e6:>7.1f}M tokens  "
                    f"{elapsed:>5.0f}s  {len(tokens) / elapsed / 1e3:>6.0f}k tok/s",
                    flush=True,
                )

    split = int(len(tokens) * (1.0 - val_fraction))
    train_stats = write_tokens(out_dir / "train.bin", tokens[:split])
    val_stats = write_tokens(out_dir / "val.bin", tokens[split:])

    unique = len(np.unique(np.asarray(tokens[: min(len(tokens), 5_000_000)])))
    return {
        "train": train_stats,
        "val": val_stats,
        "elapsed_s": time.perf_counter() - started,
        "distinct_tokens_in_first_5M": unique,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenise a JSONL corpus")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.002)
    args = parser.parse_args()

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
