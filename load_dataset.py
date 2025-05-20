import zipfile
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download

REPO_ID = "6cf/swarmbench"
DOWNLOAD_ROOT_DIR = Path("./swarmbench_dataset")
HF_TOKEN = None

DOWNLOAD_ROOT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {DOWNLOAD_ROOT_DIR.resolve()}")

try:
    zip_files_in_repo = [
        f for f in list_repo_files(REPO_ID, repo_type="dataset", token=HF_TOKEN)
        if f.endswith(".zip")
    ]

    if not zip_files_in_repo:
        print(f"No .zip files found in '{REPO_ID}'.")
    else:
        print(f"Found .zip files: {zip_files_in_repo}")

    for zip_filename in zip_files_in_repo:
        print(f"\nProcessing: {zip_filename}")
        local_zip_path = None # Initialize for potential cleanup
        try:
            # Download the file directly into the DOWNLOAD_ROOT_DIR
            downloaded_file_path_str = hf_hub_download(
                repo_id=REPO_ID,
                filename=zip_filename,
                repo_type="dataset",
                local_dir=DOWNLOAD_ROOT_DIR,
                local_dir_use_symlinks=False, # Avoid symlinks for simplicity
                token=HF_TOKEN,
            )
            local_zip_path = Path(downloaded_file_path_str)
            print(f"Downloaded: {local_zip_path.name}")

            extract_dir_name = zip_filename[:-4]
            extract_full_path = DOWNLOAD_ROOT_DIR / extract_dir_name
            extract_full_path.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_full_path)
            print(f"Extracted to: {extract_full_path.name}")

        except Exception as e:
            print(f"Error processing '{zip_filename}': {e}")
            if local_zip_path and local_zip_path.exists(): # Cleanup if download started
                try:
                    local_zip_path.unlink(missing_ok=True) # missing_ok=True (Python 3.8+)
                except TypeError: # Fallback for older Python
                    if local_zip_path.exists(): local_zip_path.unlink()

except Exception as e:
    print(f"A general error occurred: {e}")

print("\nScript finished.")