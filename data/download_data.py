import argparse
import os
from dotenv import load_dotenv
import shutil
from huggingface_hub import hf_hub_download, list_repo_files
load_dotenv()

def download_gaia(target_dir):
    print("📥 Downloading GAIA metadata.jsonl from Hugging Face...")
    repo_id = "gaia-benchmark/GAIA"
    repo_type="dataset"
    folder_prefix = "2023/"
    
    files = list_repo_files(repo_id=repo_id, repo_type=repo_type)
    target_files = [f for f in files if f.startswith(folder_prefix)]
    
    for file_path in target_files:
        print(f"Downloading: {file_path}")
        # Step 1: 下载文件
        hf_file = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type=repo_type,
        )
        
        local_path = os.path.join(target_dir, file_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy(hf_file, local_path)

        print(f"✅ File saved to: {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GAIA dataset (entire 2023 folder) from Hugging Face.")
    parser.add_argument(
        "--target",
        type=str,
        default="metadata.jsonl",
        help="Target path to save the file (e.g. data/)"
    )
    args = parser.parse_args()

    download_gaia(args.target)
    # Example use: python download_data.py --target ./data