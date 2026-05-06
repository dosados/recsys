from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm.contrib.concurrent import thread_map

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Hugging Face layout: subsamples/, metadata/ under this folder
OUTPUT_DIR = _REPO_ROOT / "datasets" / "VK-LSVD"

subsample_name = "ur0.1"

train_interactions_files = [f"subsamples/{subsample_name}/train/week_{i:02}.parquet"
                            for i in range(25)]
val_interactions_file = [f"subsamples/{subsample_name}/validation/week_25.parquet"]

metadata_files = ["metadata/users_metadata.parquet",
                  "metadata/items_metadata.parquet",
                  "metadata/item_embeddings.npz"]

ALL_FILES = (
    train_interactions_files
    + val_interactions_file
    + metadata_files
)

MAX_WORKERS = min(8, len(ALL_FILES))

REPO_ID = "deepvk/VK-LSVD"


def _download(filename: str) -> None:
    hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
        local_dir=str(OUTPUT_DIR),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    thread_map(_download, ALL_FILES, max_workers=MAX_WORKERS, desc="VK-LSVD download")


if __name__ == "__main__":
    main()
