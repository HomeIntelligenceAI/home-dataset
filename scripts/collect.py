"""
HOME AI – Dataset Collector
Downloads and prepares training data from public sources.
Supports Hugging Face datasets and local text files.
"""

import argparse
import pathlib
import json


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
