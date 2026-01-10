"""
Utilities for HuggingFace Hub integration.

This module provides functions to upload and download model checkpoints
to/from HuggingFace Hub, as well as manage authentication.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    from huggingface_hub import (
        HfApi,
        create_repo,
        upload_file,
        hf_hub_download,
        list_repo_files,
        login,
        whoami,
        Repository,
    )
    from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    HfApi = None


def check_authentication() -> tuple[bool, Optional[str]]:
    """
    Check if user is authenticated with HuggingFace Hub.
    
    Returns:
        (is_authenticated, username): Tuple indicating auth status and username if authenticated.
    """
    if not HF_AVAILABLE:
        return False, None
    
    try:
        user_info = whoami()
        username = user_info.get('name') if user_info else None
        return username is not None, username
    except Exception:
        return False, None


def authenticate(token: Optional[str] = None) -> bool:
    """
    Authenticate with HuggingFace Hub.
    
    Args:
        token: HuggingFace token. If None, will prompt for login.
    
    Returns:
        True if authentication successful, False otherwise.
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        )
    
    try:
        if token:
            login(token=token)
        else:
            login()
        return check_authentication()[0]
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("\nTo authenticate manually:")
        print("1. Get your token from https://huggingface.co/settings/tokens")
        print("2. Run: huggingface-cli login")
        print("   Or set HUGGINGFACE_HUB_TOKEN environment variable")
        return False


def ensure_repo_exists(repo_id: str, private: bool = False) -> bool:
    """
    Ensure a repository exists on HuggingFace Hub, creating it if necessary.
    
    Args:
        repo_id: Repository ID (format: username/repo-name)
        private: Whether to create a private repository
    
    Returns:
        True if repository exists or was created successfully, False otherwise.
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        )
    
    api = HfApi()
    
    try:
        # Try to get repo info - if it doesn't exist, create it
        try:
            api.repo_info(repo_id, repo_type="model")
            return True
        except RepositoryNotFoundError:
            print(f"Repository {repo_id} does not exist. Creating...")
            create_repo(repo_id=repo_id, repo_type="model", private=private)
            print(f"Created repository: {repo_id}")
            return True
    except Exception as e:
        print(f"Error ensuring repository exists: {e}")
        return False


def sanitize_repo_name(name: str) -> str:
    """
    Sanitize a repository name to be valid for HuggingFace Hub.
    
    Args:
        name: Original name
    
    Returns:
        Sanitized name (lowercase, alphanumeric + hyphens/underscores only)
    """
    # Convert to lowercase and replace invalid characters
    name = name.lower()
    # Replace spaces and special chars with hyphens
    name = re.sub(r'[^a-z0-9_-]', '-', name)
    # Remove multiple consecutive hyphens
    name = re.sub(r'-+', '-', name)
    # Remove leading/trailing hyphens
    name = name.strip('-')
    return name


def get_repo_id_from_config(config_name: str, username: Optional[str] = None) -> str:
    """
    Generate a repository ID from config name.
    
    Args:
        config_name: Name from config file
        username: HuggingFace username. If None, will try to get from auth.
    
    Returns:
        Repository ID in format username/model-name-autoencoder
    """
    if not username:
        is_auth, username = check_authentication()
        if not is_auth or not username:
            raise ValueError(
                "Cannot determine username. Please authenticate first with "
                "huggingface-cli login or set repo_id manually in config."
            )
    
    sanitized_name = sanitize_repo_name(config_name)
    repo_id = f"{username}/{sanitized_name}-autoencoder"
    return repo_id


def upload_checkpoint_to_hub(
    checkpoint_path: str,
    repo_id: str,
    filename: Optional[str] = None,
    commit_message: Optional[str] = None,
    private: bool = False,
) -> bool:
    """
    Upload a checkpoint file to HuggingFace Hub.
    
    Args:
        checkpoint_path: Local path to checkpoint file
        repo_id: Repository ID (format: username/repo-name)
        filename: Name of file in repository (defaults to basename of checkpoint_path)
        commit_message: Commit message for the upload
        private: Whether to create private repo if it doesn't exist
    
    Returns:
        True if upload successful, False otherwise.
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Check authentication
    is_auth, _ = check_authentication()
    if not is_auth:
        raise RuntimeError(
            "Not authenticated with HuggingFace Hub. "
            "Run: huggingface-cli login"
        )
    
    # Ensure repo exists
    if not ensure_repo_exists(repo_id, private=private):
        return False
    
    # Determine filename
    if filename is None:
        filename = os.path.basename(checkpoint_path)
    
    # Determine path in repo
    if filename == "best_model.pth":
        path_in_repo = filename
    else:
        # Regular checkpoints go in checkpoints/ directory
        path_in_repo = f"checkpoints/{filename}"
    
    # Generate commit message if not provided
    if commit_message is None:
        commit_message = f"Upload {filename}"
    
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        print(f"Successfully uploaded {checkpoint_path} to {repo_id}/{path_in_repo}")
        return True
    except Exception as e:
        print(f"Error uploading checkpoint: {e}")
        return False


def upload_config_to_hub(
    config_path: str,
    repo_id: str,
    commit_message: Optional[str] = None,
) -> bool:
    """
    Upload a config file to HuggingFace Hub.
    
    Args:
        config_path: Local path to config YAML file
        repo_id: Repository ID
        commit_message: Commit message
    
    Returns:
        True if upload successful, False otherwise.
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        )
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Check authentication
    is_auth, _ = check_authentication()
    if not is_auth:
        raise RuntimeError("Not authenticated with HuggingFace Hub. Run: huggingface-cli login")
    
    # Ensure repo exists
    if not ensure_repo_exists(repo_id):
        return False
    
    filename = os.path.basename(config_path)
    if commit_message is None:
        commit_message = f"Upload config: {filename}"
    
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        print(f"Successfully uploaded {config_path} to {repo_id}/{filename}")
        return True
    except Exception as e:
        print(f"Error uploading config: {e}")
        return False


def download_checkpoint_from_hub(
    repo_id: str,
    filename: str,
    local_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> Optional[str]:
    """
    Download a checkpoint file from HuggingFace Hub.
    
    Args:
        repo_id: Repository ID (format: username/repo-name)
        filename: Name of checkpoint file (e.g., "best_model.pth" or "checkpoints/checkpoint_10.pth")
        local_dir: Local directory to save file. If None, uses HuggingFace cache.
        revision: Repository revision (branch, tag, or commit). If None, uses main.
    
    Returns:
        Path to downloaded checkpoint file, or None if download failed.
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        )
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=revision,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded {filename} from {repo_id} to {downloaded_path}")
        return downloaded_path
    except EntryNotFoundError:
        print(f"File {filename} not found in repository {repo_id}")
        return None
    except Exception as e:
        print(f"Error downloading checkpoint: {e}")
        return None


def list_hub_checkpoints(repo_id: str, revision: Optional[str] = None) -> List[str]:
    """
    List all checkpoint files available in a Hub repository.
    
    Args:
        repo_id: Repository ID
        revision: Repository revision (branch, tag, or commit). If None, uses main.
    
    Returns:
        List of checkpoint filenames
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        )
    
    try:
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)
        
        # Filter for checkpoint files
        checkpoint_files = [
            f for f in files 
            if f.endswith('.pth') or f.endswith('.pt') or f.endswith('.ckpt')
        ]
        
        return checkpoint_files
    except Exception as e:
        print(f"Error listing checkpoints: {e}")
        return []


def is_hub_path(path: str) -> bool:
    """
    Check if a path is a HuggingFace Hub identifier.
    
    Args:
        path: Path to check
    
    Returns:
        True if path is a Hub identifier (format: username/repo-name or hub://username/repo-name)
    """
    # Check for hub:// prefix or username/repo-name pattern
    if path.startswith("hub://"):
        return True
    
    # Check for username/repo-name format (no spaces, has slash)
    if "/" in path and not os.path.exists(path) and not os.path.isabs(path):
        # Simple heuristic: if it looks like username/repo-name and isn't a local path
        parts = path.split("/")
        if len(parts) == 2 and all(p and not p.startswith(".") for p in parts):
            return True
    
    return False


def resolve_checkpoint_path(checkpoint_path: str, local_dir: Optional[str] = None) -> str:
    """
    Resolve a checkpoint path, downloading from Hub if necessary.
    
    Args:
        checkpoint_path: Local path or Hub identifier (username/repo-name or hub://username/repo-name)
        local_dir: Local directory to save downloaded checkpoints. If None, uses HuggingFace cache.
    
    Returns:
        Resolved local path to checkpoint file
    
    Raises:
        FileNotFoundError: If checkpoint cannot be found or downloaded
    """
    # If it's already a local path that exists, return it
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    # Check if it's a Hub path
    if is_hub_path(checkpoint_path):
        # Extract repo_id and filename
        if checkpoint_path.startswith("hub://"):
            hub_id = checkpoint_path[6:]  # Remove "hub://" prefix
        else:
            hub_id = checkpoint_path
        
        # Default to best_model.pth if no filename specified
        if "/" in hub_id and hub_id.count("/") == 1:
            repo_id = hub_id
            filename = "best_model.pth"
        else:
            # Assume format: username/repo-name/filename
            parts = hub_id.split("/")
            if len(parts) >= 3:
                repo_id = "/".join(parts[:-1])
                filename = parts[-1]
            else:
                repo_id = hub_id
                filename = "best_model.pth"
        
        # Download from Hub
        downloaded = download_checkpoint_from_hub(repo_id, filename, local_dir=local_dir)
        if downloaded is None:
            raise FileNotFoundError(
                f"Could not download checkpoint from Hub: {checkpoint_path}"
            )
        return downloaded
    
    # If it doesn't exist and isn't a Hub path, raise error
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")


def setup_hub_integration(hf_cfg: Dict, config_name: str, config_path: str, logger=None) -> Dict:
    """
    Setup HuggingFace Hub integration from config.
    
    Args:
        hf_cfg: HuggingFace config section from training config
        config_name: Experiment name for auto-generating repo_id
        config_path: Path to config file for initial upload
        logger: Optional logger for messages
        
    Returns:
        Dict with hub settings: enabled, repo_id, push_best, push_checkpoints,
        push_interval, private
    """
    result = {
        'enabled': False,
        'repo_id': hf_cfg.get('repo_id', None),
        'push_best': hf_cfg.get('push_best', True),
        'push_checkpoints': hf_cfg.get('push_checkpoints', False),
        'push_interval': hf_cfg.get('push_interval', 5),
        'private': hf_cfg.get('private', False),
    }
    
    if not hf_cfg.get('enabled', False):
        return result
    
    is_auth, username = check_authentication()
    if not is_auth:
        if logger:
            logger.warning(
                "HuggingFace Hub integration enabled but not authenticated. "
                "Run: huggingface-cli login"
            )
        return result
    
    # Auto-generate repo_id if not provided
    if result['repo_id'] is None:
        try:
            result['repo_id'] = get_repo_id_from_config(config_name, username)
        except Exception as e:
            if logger:
                logger.warning(f"Could not auto-generate repo_id: {e}. Disabling Hub integration.")
            return result
    
    result['enabled'] = True
    
    if logger:
        logger.info(f"HuggingFace Hub integration enabled. Repo ID: {result['repo_id']}")
    
    # Upload initial config
    try:
        upload_config_to_hub(config_path, result['repo_id'], commit_message="Initial config upload")
    except Exception as e:
        if logger:
            logger.warning(f"Could not upload config to Hub: {e}")
    
    return result
