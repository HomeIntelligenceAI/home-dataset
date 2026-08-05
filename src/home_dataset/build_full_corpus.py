"""Build the full compute-optimal HFM-2 corpus.

Target: 2.77B tokens — 20 per parameter for a 138M model.

R1 measured what each source actually supplies, at 0.208 tokens/char and
2.60 bytes/char for Telugu:

    FineWeb-2 tel_Telu   1.82B tokens   1,983,328 docs
    C4 te                0.47B          669,396
    Sangraha verified    0.44B          1,031,867
    IndicCorpV2          0.40B          13,498,729 (sentence-level, short)
    Wikipedia te         0.07B          87,854
    ------------------------------------------------
                         3.20B before overlap

Cross-source overlap measured at 0–7%, so ~3.0B survives — comfortably past
the 2.77B target.

Resumable: each source writes its own JSONL and is skipped if already present,
so an interrupted run continues rather than restarting.

    python -m home_dataset.build_full_corpus --out C:\\HomeAI_Corpus\\raw_full
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SOURCES", "Source", "fetch_source"]


@dataclass(frozen=True)
class Source:
    name: str
    dataset: str
    config: str | None
    split: str
    max_docs: int
    min_chars: int = 200
    text_field: str = "text"


# Ordered by tokens contributed, so an interrupted build still has the bulk.
SOURCES = [
    Source("fineweb2", "HuggingFaceFW/fineweb-2", "tel_Telu", "train", 2_000_000),
    Source("c4", "allenai/c4", "te", "train", 700_000),
    Source("sangraha", "ai4bharat/sangraha", "verified", "tel", 1_100_000),
    Source("indiccorp", "ai4bharat/IndicCorpV2", "indiccorp_v2", "tel_Telu", 14_000_000, min_chars=60),
    Source("wikipedia", "wikimedia/wikipedia", "20231101.te", "train", 100_000),
]


def fetch_source(source: Source, out_dir: Path) -> dict[str, object]:
    """Stream one source to JSONL. Skips if the file already exists."""
    from datasets import load_dataset

    path = out_dir / f"{source.name}.jsonl"
    if path.exists() and path.stat().st_size > 0:
        return {"source": source.name, "skipped": True, "bytes": path.stat().st_size}

    # Write to .part first so an interrupted download is never mistaken for a
    # complete one on the next run.
    partial = path.with_suffix(".part")
    started = time.perf_counter()
    docs = chars = 0

    try:
        stream = load_dataset(
            source.dataset, source.config, split=source.split, streaming=True
        )
    except Exception as exc:  # noqa: BLE001 - a gated or renamed source must not kill the build
        print(f"  {source.name}: UNAVAILABLE ({type(exc).__name__}) -- skipping", flush=True)
        return {"source": source.name, "error": str(exc)[:120]}

    with partial.open("w", encoding="utf-8") as handle:
        for row in stream:
            text = row.get(source.text_field) or ""
            if len(text) < source.min_chars:
                continue
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            docs += 1
            chars += len(text)
            if docs % 100_000 == 0:
                rate = docs / (time.perf_counter() - started)
                print(
                    f"  {source.name}: {docs:,} docs  {chars / 1e9:.2f}B chars  "
                    f"{rate:,.0f} docs/s",
                    flush=True,
                )
            if docs >= source.max_docs:
                break

    partial.rename(path)
    elapsed = time.perf_counter() - started
    # 0.208 tokens/char, measured in R1 on a vocab-32,768 tokenizer.
    return {
        "source": source.name,
        "docs": docs,
        "chars": chars,
        "est_tokens": int(chars * 0.208),
        "gb": path.stat().st_size / 1e9,
        "seconds": int(elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the full HFM-2 corpus")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--only", nargs="*", help="Limit to named sources (default: all)"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    selected = [s for s in SOURCES if not args.only or s.name in args.only]

    total_tokens = 0
    print(f"building corpus in {args.out}\n")
    for source in selected:
        print(f"[{source.name}] {source.dataset} / {source.config}", flush=True)
        result = fetch_source(source, args.out)
        if result.get("skipped"):
            print(f"  already present ({result['bytes'] / 1e9:.2f} GB) -- skipped\n", flush=True)
            continue
        if result.get("error"):
            print(f"  {result['error']}\n", flush=True)
            continue
        total_tokens += int(result["est_tokens"])
        print(
            f"  {result['docs']:,} docs, {result['gb']:.2f} GB, "
            f"~{int(result['est_tokens']) / 1e9:.2f}B tokens, {result['seconds']}s\n",
            flush=True,
        )

    print(f"TOTAL estimated: {total_tokens / 1e9:.2f}B tokens")
    print(f"target for 138M compute-optimal: 2.77B")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
