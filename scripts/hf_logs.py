#!/usr/bin/env python3
"""Upload / download the logs directory to a Hugging Face dataset repo.

Usage:
    # Upload logs/ to a HF dataset repo
    python hf_logs.py upload <hf_repo_id> [--logs-dir logs] [--path-in-repo logs]

    # Download from a HF dataset repo into logs/
    python hf_logs.py download <hf_repo_id> [--logs-dir logs] [--path-in-repo logs]

The repo is created automatically on first upload if it does not exist.
Authentication uses the HF_TOKEN env var or a cached `huggingface-cli login` token.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError


def upload(repo_id: str, logs_dir: Path, path_in_repo: str) -> None:
    if not logs_dir.exists():
        sys.exit(f"Error: logs directory '{logs_dir}' does not exist.")

    api = HfApi()

    # Create the repo if it doesn't exist yet
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except HfHubHTTPError as e:
        sys.exit(f"Error creating repo: {e}")

    print(f"Uploading '{logs_dir}' -> {repo_id}/{path_in_repo} ...")
    api.upload_folder(
        folder_path=str(logs_dir),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload logs from {logs_dir.resolve()}",
    )
    print("Done.")


def download(repo_id: str, logs_dir: Path, path_in_repo: str) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id}/{path_in_repo} -> '{logs_dir}' ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(logs_dir),
        allow_patterns=f"{path_in_repo}/**" if path_in_repo != "." else None,
    )
    # If path_in_repo is a subdirectory, files land inside logs_dir/path_in_repo.
    # Move them up so logs_dir matches the remote layout.
    nested = logs_dir / path_in_repo
    if path_in_repo != "." and nested.exists() and nested != logs_dir:
        for item in nested.iterdir():
            item.rename(logs_dir / item.name)
        try:
            nested.rmdir()
        except OSError:
            pass  # not empty – leave it
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync logs/ with a Hugging Face dataset repo."
    )
    parser.add_argument("command", choices=["upload", "download"])
    parser.add_argument("repo_id", help="HF repo id, e.g. 'username/stag-hunt-logs'")
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Local logs directory (default: logs)",
    )
    parser.add_argument(
        "--path-in-repo",
        default="logs",
        help="Path inside the HF repo (default: logs)",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)

    if args.command == "upload":
        upload(args.repo_id, logs_dir, args.path_in_repo)
    else:
        download(args.repo_id, logs_dir, args.path_in_repo)


if __name__ == "__main__":
    main()
