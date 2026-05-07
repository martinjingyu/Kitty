"""
Download Kitty artifacts from Hugging Face Hub.

Examples:
  python3 scripts/download_from_hf.py --raw-repo-id USER/kitty-raw-data
  python3 scripts/download_from_hf.py --model-repo-id USER/kitty-models
  python3 scripts/download_from_hf.py --raw-repo-id USER/kitty-raw-data --model-repo-id USER/kitty-models

By default existing local files are kept. Use --force to overwrite.
"""
import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models" / "saved"


def _load_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Run `pip install -r requirements.txt` first."
        ) from exc
    return snapshot_download


def _copy_tree(src: Path, dst: Path, force: bool) -> tuple[int, int]:
    copied = 0
    skipped = 0
    dst.mkdir(parents=True, exist_ok=True)

    for src_file in src.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        if dst_file.exists() and not force:
            skipped += 1
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied += 1

    return copied, skipped


def _download_folder(
    *,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    local_dir: Path,
    force: bool,
    dry_run: bool,
):
    print(f"\n{repo_type.upper()} download")
    print(f"  repo       : {repo_id}")
    print(f"  path       : {path_in_repo or '.'}")
    print(f"  local      : {local_dir}")
    print(f"  overwrite  : {'yes' if force else 'no'}")

    if dry_run:
        print("  dry-run    : skipped download")
        return

    snapshot_download = _load_snapshot_download()
    allow_patterns = f"{path_in_repo.rstrip('/')}/*" if path_in_repo else None
    snapshot_path = Path(snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        token=os.getenv("HF_TOKEN") or None,
        allow_patterns=allow_patterns,
    ))
    source = snapshot_path / path_in_repo if path_in_repo else snapshot_path
    if not source.exists():
        raise SystemExit(f"Path not found in downloaded snapshot: {path_in_repo}")

    copied, skipped = _copy_tree(source, local_dir, force=force)
    print(f"  copied     : {copied}")
    print(f"  skipped    : {skipped}")


def main():
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Download Kitty raw data and trained models from Hugging Face Hub.")
    parser.add_argument("--raw-repo-id", default=os.getenv("HF_RAW_REPO_ID"),
                        help="Dataset repo id for data/raw, e.g. USER/kitty-raw-data")
    parser.add_argument("--model-repo-id", default=os.getenv("HF_MODEL_REPO_ID"),
                        help="Model repo id for models/saved, e.g. USER/kitty-models")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help="Local raw data folder to restore")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                        help="Local trained model folder to restore")
    parser.add_argument("--raw-path-in-repo", default="raw",
                        help="Source folder inside the dataset repo")
    parser.add_argument("--model-path-in-repo", default="models",
                        help="Source folder inside the model repo")
    parser.add_argument("--raw-only", action="store_true",
                        help="Only download raw data")
    parser.add_argument("--models-only", action="store_true",
                        help="Only download trained models")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing local files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be downloaded without contacting Hugging Face")
    args = parser.parse_args()

    if args.raw_only and args.models_only:
        raise SystemExit("Choose at most one of --raw-only / --models-only.")

    want_raw = not args.models_only
    want_models = not args.raw_only

    if want_raw and not args.raw_repo_id:
        print("Skipping raw data: no --raw-repo-id or HF_RAW_REPO_ID configured.")
    elif want_raw:
        _download_folder(
            repo_id=args.raw_repo_id,
            repo_type="dataset",
            path_in_repo=args.raw_path_in_repo,
            local_dir=args.raw_dir,
            force=args.force,
            dry_run=args.dry_run,
        )

    if want_models and not args.model_repo_id:
        print("Skipping models: no --model-repo-id or HF_MODEL_REPO_ID configured.")
    elif want_models:
        _download_folder(
            repo_id=args.model_repo_id,
            repo_type="model",
            path_in_repo=args.model_path_in_repo,
            local_dir=args.model_dir,
            force=args.force,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
