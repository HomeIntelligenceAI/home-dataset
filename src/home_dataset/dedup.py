"""Near-duplicate detection with MinHash + LSH.

Exact-hash deduplication (see :mod:`home_dataset.prepare`) only catches
byte-identical documents. Web corpora are full of documents that differ by a
timestamp, a cookie banner, or one navigation link, and those are just as
useless for training — the model sees the same text many times and
over-weights it.

MinHash estimates Jaccard similarity between shingle sets in constant space per
document, and LSH banding avoids the O(n^2) comparison that would make it
unusable at corpus scale.

Implemented here rather than pulled from `datasketch` for the same reason the
tokenizer and transformer are: the project builds its own pipeline. It is ~150
lines and the algorithm is worth understanding when the dedup rate turns out to
decide the model size.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

__all__ = ["DedupStats", "MinHasher", "deduplicate_near", "shingle"]

# 64-bit Mersenne prime, standard choice for the universal-hash family below.
_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1

_WORD = re.compile(r"\S+")


def shingle(text: str, n: int = 5) -> set[int]:
    """Hash the document's word n-grams into 32-bit ints.

    Word shingles rather than character shingles: Telugu words are long and
    character shingles would make near-identical documents look distinct
    wherever a single akshara differs.
    """
    words = _WORD.findall(text)
    if len(words) < n:
        # Short documents shingle as a single unit rather than vanishing.
        return {int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), "big")}
    return {
        int.from_bytes(
            hashlib.blake2b(" ".join(words[i : i + n]).encode(), digest_size=4).digest(),
            "big",
        )
        for i in range(len(words) - n + 1)
    }


@dataclass(frozen=True)
class DedupStats:
    documents_in: int
    documents_kept: int
    exact_duplicates: int
    near_duplicates: int

    @property
    def removed_fraction(self) -> float:
        if not self.documents_in:
            return 0.0
        return 1.0 - self.documents_kept / self.documents_in

    def __str__(self) -> str:
        return (
            f"{self.documents_kept:,} kept of {self.documents_in:,} "
            f"({self.removed_fraction:.1%} removed: "
            f"{self.exact_duplicates:,} exact, {self.near_duplicates:,} near)"
        )


class MinHasher:
    """Fixed-width MinHash signatures over shingle sets.

    Uses the universal family h(x) = (a*x + b) mod p, one (a, b) pair per
    permutation. Seeded, so signatures are reproducible across runs — a dedup
    pass that gives different answers each time cannot be used as a gate.
    """

    def __init__(self, num_perm: int = 128, seed: int = 1337):
        self.num_perm = num_perm
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _PRIME, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _PRIME, size=num_perm, dtype=np.uint64)

    def signature(self, shingles: set[int]) -> np.ndarray:
        if not shingles:
            return np.full(self.num_perm, _MAX_HASH, dtype=np.uint64)
        values = np.fromiter(shingles, dtype=np.uint64, count=len(shingles))
        # (n_shingles, num_perm) then min down the shingle axis.
        hashed = (np.outer(values, self.a) + self.b) % _PRIME
        return hashed.min(axis=0).astype(np.uint64)

    @staticmethod
    def similarity(left: np.ndarray, right: np.ndarray) -> float:
        """Estimated Jaccard similarity: the fraction of matching positions."""
        return float(np.count_nonzero(left == right) / len(left))


def _bands(signature: np.ndarray, num_bands: int) -> Iterator[bytes]:
    """Split a signature into bands, each hashed to one bucket key.

    Two documents become candidates if *any* band matches exactly. With b bands
    of r rows, the probability of becoming a candidate at similarity s is
    1 - (1 - s^r)^b — a sharp threshold near s = (1/b)^(1/r).
    """
    rows = len(signature) // num_bands
    for i in range(num_bands):
        band = signature[i * rows : (i + 1) * rows]
        yield hashlib.blake2b(band.tobytes(), digest_size=8).digest()


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[max(rx, ry)] = min(rx, ry)


def deduplicate_near(
    documents: Iterable[str],
    threshold: float = 0.8,
    num_perm: int = 128,
    num_bands: int = 16,
    shingle_size: int = 5,
    seed: int = 1337,
) -> tuple[list[str], DedupStats]:
    """Drop exact and near-duplicate documents, keeping the first of each cluster.

    ``threshold`` is estimated Jaccard similarity over word shingles. 0.8 is the
    common choice for web text: high enough that genuinely different documents
    on the same topic survive, low enough to catch boilerplate variants.
    """
    docs = list(documents)
    stats_in = len(docs)

    # Exact pass first -- far cheaper, and removes the bulk on web corpora.
    seen: set[str] = set()
    unique: list[str] = []
    exact = 0
    for doc in docs:
        digest = hashlib.blake2b(doc.strip().encode("utf-8"), digest_size=16).hexdigest()
        if digest in seen:
            exact += 1
            continue
        seen.add(digest)
        unique.append(doc)

    hasher = MinHasher(num_perm=num_perm, seed=seed)
    signatures = [hasher.signature(shingle(d, shingle_size)) for d in unique]

    buckets: dict[bytes, list[int]] = defaultdict(list)
    for index, signature in enumerate(signatures):
        for key in _bands(signature, num_bands):
            buckets[key].append(index)

    union = _UnionFind(len(unique))
    for members in buckets.values():
        if len(members) < 2:
            continue
        # Verify candidates against the actual signature; banding is designed
        # to over-generate, and keeping false positives would delete real text.
        first = members[0]
        for other in members[1:]:
            if hasher.similarity(signatures[first], signatures[other]) >= threshold:
                union.union(first, other)

    kept_indices = sorted({union.find(i) for i in range(len(unique))})
    kept = [unique[i] for i in kept_indices]

    return kept, DedupStats(
        documents_in=stats_in,
        documents_kept=len(kept),
        exact_duplicates=exact,
        near_duplicates=len(unique) - len(kept),
    )
