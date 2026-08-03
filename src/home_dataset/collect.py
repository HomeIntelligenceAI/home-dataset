"""
HOME AI – Dataset Collector
Downloads and prepares training data from public sources.
Supports Hugging Face datasets and local text files.
"""

import argparse
import json
import pathlib
import urllib.parse
import urllib.request

# Wikimedia rejects the default urllib agent with 403. A descriptive agent is
# their documented requirement, not a workaround.
USER_AGENT = "HomeAI-dataset/0.1 (https://github.com/HomeIntelligenceAI/home-dataset)"


def _request(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def load_from_huggingface(dataset_name: str, split: str = "train",
                          max_samples: int = 10000,
                          output_dir: str = "data") -> pathlib.Path:
    """Download a dataset from Hugging Face and save as JSONL.

    Args:
        dataset_name: e.g. "wikitext/wikitext-103-raw-v1"
        split: dataset split to use.
        max_samples: cap the number of rows.
        output_dir: where to save the output file.

    Returns:
        Path to the saved JSONL file.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install `datasets` to use HF loader: pip install datasets")

    ds = load_dataset(dataset_name, split=split, streaming=True)
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_name = dataset_name.replace("/", "_")
    out_file = out / f"{safe_name}_{split}.jsonl"

    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "")
            if not text or len(text.strip()) < 10:
                continue
            f.write(json.dumps({"text": text.strip()}, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_samples:
                break

    print(f"[Dataset] Saved {count} samples to {out_file}")
    return out_file


def fetch_url(url: str, destination: pathlib.Path, timeout: int = 60) -> pathlib.Path:
    """Download a plain-text file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_request(url, timeout=timeout))
    return destination


def fetch_wikipedia(lang: str, limit: int = 200, min_chars: int = 500,
                    timeout: int = 30):
    """Yield article extracts from a Wikipedia edition.

    generator=random rather than a fixed list, so repeat runs widen coverage
    instead of re-fetching the same head articles.
    """
    endpoint = f"https://{lang}.wikipedia.org/w/api.php"
    fetched = 0
    # The API caps extracts at 20 pages per request regardless of what is asked.
    while fetched < limit:
        params = {
            "action": "query", "format": "json",
            "generator": "random", "grnnamespace": "0", "grnlimit": "20",
            "prop": "extracts", "explaintext": "1", "exlimit": "20",
        }
        url = f"{endpoint}?{urllib.parse.urlencode(params)}"
        try:
            payload = json.loads(_request(url, timeout=timeout))
        except Exception:  # a flaky page must not kill a long crawl
            break
        pages = payload.get("query", {}).get("pages", {})
        if not pages:
            break
        for page in pages.values():
            extract = (page.get("extract") or "").strip()
            if len(extract) >= min_chars:
                yield extract
                fetched += 1
                if fetched >= limit:
                    return


def load_local_text(text_dir: str, output_dir: str = "data") -> pathlib.Path:
    """Read all .txt files from a directory and merge into a JSONL file."""
    src = pathlib.Path(text_dir)
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "local_text.jsonl"

    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for txt_file in sorted(src.glob("**/*.txt")):
            text = txt_file.read_text(encoding="utf-8", errors="replace").strip()
            if len(text) < 10:
                continue
            f.write(json.dumps({"text": text, "source": str(txt_file)}, ensure_ascii=False) + "\n")
            count += 1

    print(f"[Dataset] Merged {count} text files into {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="HOME AI dataset collector")
    sub = parser.add_subparsers(dest="command")

    hf = sub.add_parser("huggingface", help="Download from Hugging Face")
    hf.add_argument("--dataset", type=str, default="wikitext",
                    help="HF dataset name (e.g. wikitext)")
    hf.add_argument("--split", type=str, default="train")
    hf.add_argument("--max-samples", type=int, default=10000)
    hf.add_argument("--output-dir", type=str, default="data")

    local = sub.add_parser("local", help="Merge local text files")
    local.add_argument("--text-dir", type=str, required=True)
    local.add_argument("--output-dir", type=str, default="data")

    args = parser.parse_args()

    if args.command == "huggingface":
        load_from_huggingface(args.dataset, args.split, args.max_samples, args.output_dir)
    elif args.command == "local":
        load_local_text(args.text_dir, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
