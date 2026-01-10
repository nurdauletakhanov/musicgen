"""
Standalone script to upload existing checkpoints to HuggingFace Hub.

This script is useful for uploading models that were trained before
HuggingFace Hub integration was added.

Usage:
    # Upload a single checkpoint
    python scripts/upload_to_hub.py --checkpoint checkpoints/stft-vocoder/best_model.pth --repo-id username/model-name

    # Upload multiple checkpoints from a directory
    python scripts/upload_to_hub.py --checkpoint-dir checkpoints/stft-vocoder --repo-id username/model-name

    # Upload config file as well
    python scripts/upload_to_hub.py --checkpoint checkpoints/stft-vocoder/best_model.pth --repo-id username/model-name --config config.yaml

    # Upload all checkpoints in a directory (including best_model.pth)
    python scripts/upload_to_hub.py --checkpoint-dir checkpoints/stft-vocoder --repo-id username/model-name --upload-all
"""

import os
import argparse
import glob
from pathlib import Path

from training.hub_utils import (
    check_authentication,
    authenticate,
    ensure_repo_exists,
    upload_checkpoint_to_hub,
    upload_config_to_hub,
)


def main():
    parser = argparse.ArgumentParser(
        description="Upload checkpoints to HuggingFace Hub"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a single checkpoint file to upload"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory containing checkpoints to upload"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (format: username/repo-name)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file to upload"
    )
    parser.add_argument(
        "--upload-all",
        action="store_true",
        help="Upload all checkpoints in directory (including best_model.pth). "
             "If False, only uploads best_model.pth"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository if it doesn't exist"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace token (if not already authenticated)"
    )
    
    args = parser.parse_args()
    
    # Check authentication
    is_auth, username = check_authentication()
    if not is_auth:
        print("Not authenticated with HuggingFace Hub.")
        if args.token:
            print("Authenticating with provided token...")
            if authenticate(token=args.token):
                print("Authentication successful!")
            else:
                print("Authentication failed. Exiting.")
                return
        else:
            print("Please authenticate first:")
            print("1. Run: huggingface-cli login")
            print("2. Or set HUGGINGFACE_HUB_TOKEN environment variable")
            print("3. Or pass --token YOUR_TOKEN")
            return
    else:
        print(f"Authenticated as: {username}")
    
    # Ensure repository exists
    print(f"Ensuring repository exists: {args.repo_id}")
    if not ensure_repo_exists(args.repo_id, private=args.private):
        print(f"Failed to create/access repository: {args.repo_id}")
        return
    
    uploaded_files = []
    failed_files = []
    
    # Upload config if provided
    if args.config:
        if os.path.exists(args.config):
            print(f"Uploading config: {args.config}")
            if upload_config_to_hub(
                args.config,
                args.repo_id,
                commit_message="Upload training config"
            ):
                uploaded_files.append(args.config)
            else:
                failed_files.append(args.config)
        else:
            print(f"Config file not found: {args.config}")
            failed_files.append(args.config)
    
    # Upload checkpoints
    if args.checkpoint:
        # Single checkpoint file
        if os.path.exists(args.checkpoint):
            filename = os.path.basename(args.checkpoint)
            print(f"Uploading checkpoint: {args.checkpoint}")
            if upload_checkpoint_to_hub(
                args.checkpoint,
                args.repo_id,
                filename=filename,
                commit_message=f"Upload {filename}",
                private=args.private,
            ):
                uploaded_files.append(args.checkpoint)
            else:
                failed_files.append(args.checkpoint)
        else:
            print(f"Checkpoint file not found: {args.checkpoint}")
            failed_files.append(args.checkpoint)
    
    elif args.checkpoint_dir:
        # Multiple checkpoints from directory
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            print(f"Checkpoint directory not found: {args.checkpoint_dir}")
            return
        
        if args.upload_all:
            # Upload all .pth files
            checkpoint_files = list(checkpoint_dir.glob("*.pth"))
            print(f"Found {len(checkpoint_files)} checkpoint files to upload")
        else:
            # Only upload best_model.pth
            best_model_path = checkpoint_dir / "best_model.pth"
            if best_model_path.exists():
                checkpoint_files = [best_model_path]
                print("Uploading best_model.pth only (use --upload-all to upload all checkpoints)")
            else:
                print(f"best_model.pth not found in {args.checkpoint_dir}")
                checkpoint_files = []
        
        for checkpoint_path in checkpoint_files:
            filename = checkpoint_path.name
            print(f"Uploading checkpoint: {checkpoint_path}")
            if upload_checkpoint_to_hub(
                str(checkpoint_path),
                args.repo_id,
                filename=filename,
                commit_message=f"Upload {filename}",
                private=args.private,
            ):
                uploaded_files.append(str(checkpoint_path))
            else:
                failed_files.append(str(checkpoint_path))
    else:
        parser.error("Must provide either --checkpoint or --checkpoint-dir")
    
    # Print summary
    print("\n" + "="*60)
    print("Upload Summary")
    print("="*60)
    print(f"Repository: {args.repo_id}")
    print(f"Successfully uploaded: {len(uploaded_files)} file(s)")
    for f in uploaded_files:
        print(f"  ✓ {os.path.basename(f)}")
    
    if failed_files:
        print(f"\nFailed to upload: {len(failed_files)} file(s)")
        for f in failed_files:
            print(f"  ✗ {os.path.basename(f)}")
    
    if uploaded_files:
        print(f"\nView your repository at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

