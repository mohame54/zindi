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
