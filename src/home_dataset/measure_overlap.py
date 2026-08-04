"""Estimate how much two corpora actually overlap.

Naive sampling cannot answer this. If you compare 200 documents from corpus B
against 8,000 from corpus A, a genuine duplicate pair has roughly a
(8,000 / |A|) chance of both halves landing in the comparison — so with A in the
millions you measure zero overlap whether the true figure is 0% or 60%.

The fix is to index as much of A as you can afford, record what fraction of A
that represents, and scale the observed hit rate by it. The result is still an
estimate, but a defensible one with a stated multiplier rather than a
meaningless zero.

Run as:
    python -m home_dataset.measure_overlap --index corpus.jsonl --against a.txt b.txt
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from home_dataset.dedup import MinHasher, _bands, shingle

__all__ = ["OverlapIndex"]


class OverlapIndex:
    """An LSH index over one corpus, queryable by documents from another."""

    def __init__(self, num_perm: int = 128, num_bands: int = 16, seed: int = 1337):
        self.hasher = MinHasher(num_perm=num_perm, seed=seed)
        self.num_bands = num_bands
        self.buckets: dict[bytes, list[int]] = defaultdict(list)
        self.signatures: list = []

    def add(self, text: str) -> None:
        signature = self.hasher.signature(shingle(text))
        index = len(self.signatures)
        self.signatures.append(signature)
        for key in _bands(signature, self.num_bands):
            self.buckets[key].append(index)

    def matches(self, text: str, threshold: float = 0.8) -> bool:
        signature = self.hasher.signature(shingle(text))
        candidates = {
            c for key in _bands(signature, self.num_bands) for c in self.buckets.get(key, ())
        }
        return any(
            self.hasher.similarity(signature, self.signatures[c]) >= threshold
            for c in candidates
        )

    def __len__(self) -> int:
        return len(self.signatures)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate cross-corpus overlap")
    parser.add_argument("--index", type=Path, required=True, help="JSONL to index")
    parser.add_argument("--against", type=Path, nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=150_000)
    parser.add_argument("--population", type=int, required=True,
                        help="Total docs in the indexed corpus, for the coverage scale")
    parser.add_argument("--probe", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    index = OverlapIndex()
    start = time.perf_counter()
    with args.index.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= args.limit:
                break
            index.add(json.loads(line)["text"])
            if (i + 1) % 25_000 == 0:
                print(f"  indexed {i + 1:,} in {time.perf_counter() - start:.0f}s", flush=True)

    coverage = len(index) / args.population
    print(
        f"INDEX {len(index):,} docs, {len(index.buckets):,} buckets, "
        f"{time.perf_counter() - start:.0f}s, covers {100 * coverage:.1f}% of corpus",
        flush=True,
    )
    print(f"\n{'source':<14} {'probed':>7} {'hit':>5} {'raw':>7} {'est. true overlap':>19}", flush=True)

    for path in args.against:
        docs = [d for d in path.read_text(encoding="utf-8").split("\n\n") if len(d) > 200]
        docs = docs[: args.probe]
        if not docs:
            continue
        hits = sum(index.matches(d, args.threshold) for d in docs)
        raw = hits / len(docs)
        est = min(1.0, raw / coverage) if coverage else 0.0
        print(
            f"{path.stem[:14]:<14} {len(docs):>7} {hits:>5} {100 * raw:>6.1f}% {100 * est:>18.1f}%",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
