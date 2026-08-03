"""Turn raw text into the token stream the trainer reads.

    text files -> clean -> deduplicate -> tokenise -> train.bin / val.bin

The output is a flat uint16 stream (see home_training.data). Everything here is
offline once the raw text is on disk; the only step that touches the network is
the optional Hugging Face download in ``collect.py``.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

__all__ = ["CleanStats", "clean_text", "deduplicate", "iter_documents", "prepare"]

# Telugu occupies U+0C00-U+0C7F. Keeping the ranges explicit means a corpus
# that has silently become all-English is visible in the stats rather than
# discovered after a training run.
TELUGU = re.compile(r"[ఀ-౿]")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WHITESPACE = re.compile(r"[ \t]+")
BLANK_LINES = re.compile(r"\n{3,}")


@dataclass
class CleanStats:
    documents_in: int = 0
    documents_kept: int = 0
    duplicates: int = 0
    too_short: int = 0
    telugu_docs: int = 0
    characters: int = 0

    @property
    def telugu_share(self) -> float:
        return self.telugu_docs / self.documents_kept if self.documents_kept else 0.0

    def __str__(self) -> str:
        return (
            f"{self.documents_kept:,} kept of {self.documents_in:,} "
            f"({self.duplicates:,} duplicate, {self.too_short:,} too short)\n"
            f"{self.characters:,} characters, "
            f"{self.telugu_share:.1%} of documents contain Telugu"
        )


def clean_text(text: str) -> str:
    """Normalise a document without destroying script information.

    NFC composition matters for Telugu: the same akshara can be encoded as a
    composed codepoint or as base plus vowel sign, and a tokeniser trained on
    one form will not recognise the other.
    """
    text = unicodedata.normalize("NFC", text)
    text = CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return BLANK_LINES.sub("\n\n", text).strip()


def iter_documents(paths: Iterable[Path]) -> Iterator[str]:
    """Yield documents from .txt (whole file) and .jsonl (one 'text' per line)."""
    import json

    for path in paths:
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if text := record.get("text"):
                        yield text
        else:
            yield path.read_text(encoding="utf-8", errors="replace")


def deduplicate(
    documents: Iterable[str], min_chars: int = 64
) -> tuple[list[str], CleanStats]:
    """Clean, drop near-empties, and remove exact duplicates.

    Exact-hash dedup only. Near-duplicate detection (MinHash) is worth adding
    once the corpus is large enough to justify it; scraped corpora are usually
    5-30% exact duplicates, and that is the cheap majority of the win.
    """
    stats = CleanStats()
    seen: set[str] = set()
    kept: list[str] = []

    for raw in documents:
        stats.documents_in += 1
        text = clean_text(raw)
        if len(text) < min_chars:
            stats.too_short += 1
            continue
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
        if digest in seen:
            stats.duplicates += 1
            continue
        seen.add(digest)
        kept.append(text)
        stats.documents_kept += 1
        stats.characters += len(text)
        if TELUGU.search(text):
            stats.telugu_docs += 1

    return kept, stats


def prepare(
    sources: Iterable[Path],
    out_dir: Path,
    tokenizer,  # noqa: ANN001 - home_tokenizer.HomeTokenizer, imported by the caller
    val_fraction: float = 0.005,
    min_chars: int = 64,
    eos_id: int | None = None,
) -> dict[str, object]:
    """Clean, tokenise, and write train.bin / val.bin.

    Documents are joined by an end-of-text token rather than concatenated
    blindly, so the model learns where a document stops instead of learning to
    run one into the next.
    """
    from home_training.data import write_tokens

    documents, stats = deduplicate(iter_documents(sources), min_chars=min_chars)
    if not documents:
        raise ValueError("no documents survived cleaning")

    tokens: list[int] = []
    for document in documents:
        tokens.extend(tokenizer.encode(document))
        if eos_id is not None:
            tokens.append(eos_id)

    # Split by position rather than by document: the validation set should be
    # held-out text, and a random document split leaks near-duplicates across
    # the boundary far more easily.
    split = int(len(tokens) * (1.0 - val_fraction))
    out_dir.mkdir(parents=True, exist_ok=True)
    train_stats = write_tokens(out_dir / "train.bin", tokens[:split])
    val_stats = write_tokens(out_dir / "val.bin", tokens[split:])

    return {
        "clean": stats,
        "train": train_stats,
        "val": val_stats,
        "tokens_per_char": len(tokens) / max(1, stats.characters),
    }
