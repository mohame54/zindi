import os
from huggingface_hub import HfApi


def load_hf_api():
   from huggingface_hub.hf_api import HfFolder
   HfFolder.save_token(os.getenv("HF_TOKEN"))
   return HfApi()


def upload_file_paths_to_hf(pathes):
    api = load_hf_api()
    repo_id = os.getenv("HF_REPO_ID")
    for pth in pathes:
         api.upload_file(
            path_or_fileobj= pth,
            path_in_repo =pth,
            repo_id=repo_id,
            repo_type="model",
        )


def upload_folder_to_hf(
    local_dir: str,
    path_in_repo: str = "",
    repo_id: str | None = None,
    commit_message: str = "Upload model checkpoint",
) -> str:
    """Push an entire local directory to a HuggingFace Hub model repo.

    Args:
        local_dir: Local folder to upload (e.g. ``output_dir``).
        path_in_repo: Sub-folder inside the HF repo (``""`` = repo root).
        repo_id: HF repo id; falls back to ``HF_REPO_ID`` env var.
        commit_message: Commit message shown in the HF repo history.

    Returns:
        The URL of the uploaded folder on HF Hub.
    """
    api = load_hf_api()
    repo_id = repo_id or os.getenv("HF_REPO_ID")
    if not repo_id:
        raise ValueError(
            "No HF repo id: pass repo_id or set the HF_REPO_ID environment variable."
        )
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    url = api.upload_folder(
        folder_path=local_dir,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"Model pushed to: {url}")
    return url

def download_checkpoint_from_hf(checkpoint_dir, local_dir, pathes):
    from huggingface_hub import hf_hub_download
    
    os.makedirs(local_dir, exist_ok=True)
    repo_id = os.getenv("HF_REPO_ID")
    
    print(f"Downloading checkpoint from {repo_id}/{checkpoint_dir} to {local_dir}...")
    
    for filename in pathes:
        hf_hub_download(
            repo_id=repo_id,
            filename=f"{checkpoint_dir}/{filename}",
            local_dir=local_dir,
        )
        print(f"Downloaded {filename}")
    return os.path.join(local_dir, checkpoint_dir)
