import os
import shutil
import pandas as pd
from tqdm import tqdm
import kagglehub

# === Configuration ===
DATA_DIR = os.path.join("..", "data", "raw")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
METADATA_FILE = os.path.join(DATA_DIR, "styles.csv")
DATASET_NAME = "paramaggarwal/fashion-product-images-dataset"

# === Download Dataset ===
def download_with_kagglehub():
    print("📥 Downloading dataset using KaggleHub...")
    dataset_path = kagglehub.dataset_download(DATASET_NAME)
    full_path = os.path.join(dataset_path, "fashion-dataset", "fashion-dataset")
    print(f"✅ Dataset downloaded to: {full_path}")
    return full_path

# === Save Metadata Locally ===
def prepare_metadata(dataset_path):
    metadata_path = os.path.join(dataset_path, "styles.csv")
    df = pd.read_csv(metadata_path, on_bad_lines='skip')
    df['filename'] = df['id'].astype(str) + ".jpg"
    df = df.dropna(subset=['filename', 'articleType'])

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(METADATA_FILE, index=False)
    print(f"✅ Saved metadata to {METADATA_FILE}")
    return df, os.path.join(dataset_path, "images")

# === Save Images Locally ===
def copy_images(df, source_image_dir):
    os.makedirs(IMAGE_DIR, exist_ok=True)
    print(f"📦 Copying image files to {IMAGE_DIR}...")
    copied = 0
    for filename in tqdm(df['filename'].unique()):
        src = os.path.join(source_image_dir, filename)
        dst = os.path.join(IMAGE_DIR, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
    print(f"✅ Copied {copied} image files.")

def main():
    dataset_path = download_with_kagglehub()
    df, image_dir = prepare_metadata(dataset_path)
    copy_images(df, image_dir)

if __name__ == "__main__":
    main()
