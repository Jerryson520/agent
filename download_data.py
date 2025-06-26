import argparse
import os
from dotenv import load_dotenv
import shutil
from huggingface_hub import hf_hub_download
load_dotenv()

def download_gaia(target_path):
    print("📥 Downloading GAIA metadata.jsonl from Hugging Face...")
    
    # Step 1: 下载文件
    hf_path = hf_hub_download(
        repo_id="gaia-benchmark/GAIA",
        filename="2023/validation/metadata.jsonl",
        repo_type="dataset"
    )
    # Step 2: 创建目标文件夹（如果不存在）
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Step 3: 复制到指定路径
    shutil.copy(hf_path, target_path)

    print(f"✅ File saved to: {target_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GAIA metadata.jsonl from Hugging Face.")
    parser.add_argument(
        "--target",
        type=str,
        default="metadata.jsonl",
        help="Target path to save the file (e.g. data/metadata.jsonl)"
    )
    args = parser.parse_args()

    download_gaia(args.target)
    # Example use: python3 download_data.py --target /metadata.jsonl