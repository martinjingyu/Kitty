"""
Upload Kitty artifacts to Hugging Face Hub.

Examples:
  python3 scripts/upload_to_hf.py --raw-repo-id USER/kitty-raw-data
  python3 scripts/upload_to_hf.py --model-repo-id USER/kitty-models
  python3 scripts/upload_to_hf.py --raw-repo-id USER/kitty-raw-data --model-repo-id USER/kitty-models --private

Authentication:
  - Set HF_TOKEN in .env / environment, or
  - Run `hf auth login` once on the machine.
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models" / "saved"


def _load_hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Run `pip install -r requirements.txt` first."
        ) from exc
    return HfApi


def _upload_folder(
    *,
    folder_path: Path,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    private: bool,
    commit_message: str,
    dry_run: bool,
):
    if not folder_path.exists():
        raise SystemExit(f"Folder not found: {folder_path}")

    files = [p for p in folder_path.rglob("*") if p.is_file()]
    if not files:
        raise SystemExit(f"No files to upload in: {folder_path}")

    print(f"\n{repo_type.upper()} upload")
    print(f"  local      : {folder_path}")
    print(f"  repo       : {repo_id}")
    print(f"  path       : {path_in_repo or '.'}")
    print(f"  files      : {len(files)}")
    print(f"  visibility : {'private' if private else 'public/existing'}")

    if dry_run:
        print("  dry-run    : skipped upload")
        return

    HfApi = _load_hf_api()
    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        ignore_patterns=[
            "__pycache__/**",
            "*.tmp",
            "*.log",
            ".DS_Store",
        ],
    )
    print("  status     : uploaded")


def main():
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Upload Kitty raw data and trained models to Hugging Face Hub.")
    parser.add_argument("--raw-repo-id", default=os.getenv("HF_RAW_REPO_ID"),
                        help="Dataset repo id for data/raw, e.g. USER/kitty-raw-data")
    parser.add_argument("--model-repo-id", default=os.getenv("HF_MODEL_REPO_ID"),
                        help="Model repo id for models/saved, e.g. USER/kitty-models")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help="Local raw data folder to upload")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                        help="Local trained model folder to upload")
    parser.add_argument("--raw-path-in-repo", default="raw",
                        help="Destination folder inside the dataset repo")
    parser.add_argument("--model-path-in-repo", default="models",
                        help="Destination folder inside the model repo")
    parser.add_argument("--private", action="store_true",
                        help="Create repos as private if they do not already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be uploaded without contacting Hugging Face")
    args = parser.parse_args()

    if not args.raw_repo_id and not args.model_repo_id:
        raise SystemExit("Provide --raw-repo-id and/or --model-repo-id, or set HF_RAW_REPO_ID/HF_MODEL_REPO_ID.")

    if args.raw_repo_id:
        _upload_folder(
            folder_path=args.raw_dir,
            repo_id=args.raw_repo_id,
            repo_type="dataset",
            path_in_repo=args.raw_path_in_repo,
            private=args.private,
            commit_message="Upload Kitty raw market data",
            dry_run=args.dry_run,
        )

    if args.model_repo_id:
        _upload_folder(
            folder_path=args.model_dir,
            repo_id=args.model_repo_id,
            repo_type="model",
            path_in_repo=args.model_path_in_repo,
            private=args.private,
            commit_message="Upload Kitty trained models",
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
