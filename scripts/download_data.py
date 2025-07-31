import os
import zipfile
import requests
from tqdm import tqdm

DEST_DIR = "data"
ZIP_NAME = "dataset.zip"
GOOGLE_DRIVE_FILE_ID = "1qY3mOd_fZ2XeBMGrDqEX7HasARWyOnp7"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}&export=download"
def download_file_from_google_drive(url, dest_path):
    print("Downloading dataset...")
    session = requests.Session()
    response = session.get(url, stream=True)

    with open(dest_path, "wb") as f:
        total_size = int(response.headers.get('content-length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def extract_zip(zip_path, extract_to):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def main():
    if os.path.exists(DEST_DIR):
        print(f"Dataset folder '{DEST_DIR}' already exists. Skipping download.")
        return

    os.makedirs("tmp", exist_ok=True)
    zip_path = os.path.join("tmp", ZIP_NAME)

    download_file_from_google_drive(DOWNLOAD_URL, zip_path)
    extract_zip(zip_path, DEST_DIR)

    print("✅ Done.")
    os.remove(zip_path)
    os.rmdir("tmp")

if __name__ == "__main__":
    main()
