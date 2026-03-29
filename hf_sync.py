"""HuggingFace Hub sync for persistent checkpoint & log storage.

Handles uploading checkpoints/logs to HF Hub and downloading them back.
Designed for ephemeral environments like Kaggle where local storage is lost.

Usage:
    from hf_sync import HFSync
    sync = HFSync("username/alpha-llm-checkpoints")
    sync.upload_checkpoint("checkpoints/iter_5.pt")
    sync.download_checkpoint("iter_5.pt", "checkpoints/iter_5.pt")
"""

import os
from typing import Optional

import logger


class HFSync:
    """Sync checkpoints and logs to/from HuggingFace Hub."""

    def __init__(self, repo_id: str, token: Optional[str] = None):
        """
        Args:
            repo_id: HF repo like "username/alpha-llm-checkpoints"
            token: HF token (defaults to HF_TOKEN env var or cached login)
        """
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self._api = None
        self._ensure_repo()

    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
        return self._api

    def _ensure_repo(self):
        """Create the repo if it doesn't exist."""
        try:
            from huggingface_hub import create_repo
            create_repo(self.repo_id, repo_type="model", exist_ok=True, token=self.token)
        except Exception as e:
            logger.console.print(f"[yellow]  HF repo setup: {e}[/]")

    def upload_file(self, local_path: str, repo_path: Optional[str] = None):
        """Upload a single file to the repo.

        Args:
            local_path: path to local file
            repo_path: path inside the repo (defaults to filename)
        """
        if not os.path.exists(local_path):
            return
        repo_path = repo_path or os.path.basename(local_path)
        try:
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=self.repo_id,
                repo_type="model",
            )
            logger.console.print(f"  [dim]Uploaded to HF: {repo_path}[/]")
        except Exception as e:
            logger.console.print(f"[yellow]  HF upload failed ({repo_path}): {e}[/]")

    def upload_checkpoint(self, local_path: str):
        """Upload a checkpoint .pt file."""
        filename = os.path.basename(local_path)
        self.upload_file(local_path, f"checkpoints/{filename}")

    def upload_log(self, local_path: str):
        """Upload a log file."""
        filename = os.path.basename(local_path)
        self.upload_file(local_path, f"logs/{filename}")

    def download_file(self, repo_path: str, local_path: str) -> bool:
        """Download a file from the repo.

        Returns True if successful, False otherwise.
        """
        try:
            from huggingface_hub import hf_hub_download
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                filename=repo_path,
                repo_type="model",
                local_dir=os.path.dirname(local_path),
                token=self.token,
            )
            # hf_hub_download may place the file in a subfolder matching repo_path
            # Move it to the exact local_path if needed
            if os.path.exists(downloaded) and os.path.abspath(downloaded) != os.path.abspath(local_path):
                import shutil
                shutil.move(downloaded, local_path)
            logger.console.print(f"  [dim]Downloaded from HF: {repo_path}[/]")
            return True
        except Exception as e:
            logger.console.print(f"[yellow]  HF download failed ({repo_path}): {e}[/]")
            return False

    def download_checkpoint(self, checkpoint_name: str, local_path: str) -> bool:
        """Download a checkpoint from HF.

        Args:
            checkpoint_name: e.g. "iter_5.pt" or "final.pt"
            local_path: where to save locally
        """
        return self.download_file(f"checkpoints/{checkpoint_name}", local_path)

    def download_log(self, log_name: str, local_path: str) -> bool:
        """Download a log file from HF."""
        return self.download_file(f"logs/{log_name}", local_path)

    def list_checkpoints(self) -> list:
        """List available checkpoints in the repo."""
        try:
            files = self.api.list_repo_files(self.repo_id, repo_type="model")
            return [f for f in files if f.startswith("checkpoints/") and f.endswith(".pt")]
        except Exception:
            return []

    def get_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint by iteration number."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        # Parse iteration numbers, prefer "final.pt" as highest priority
        best = None
        best_iter = -1
        for ckpt in checkpoints:
            name = os.path.basename(ckpt)
            if name == "final.pt":
                return ckpt
            if name.startswith("iter_") and name.endswith(".pt"):
                try:
                    num = int(name[5:-3])
                    if num > best_iter:
                        best_iter = num
                        best = ckpt
                except ValueError:
                    pass
        return best
