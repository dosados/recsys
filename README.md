# User Embedding Aggregator (RecSys)


This project computes user preferences from user–item interactions and item embeddings. It requires an `.npz` file of item embeddings and Parquet tables of scored user–item interactions. User preferences are computed via an Exponential Moving Average. The program stores the result as an `npz` file with arrays `(user_id, embedding)`. It also creates a `data/` folder for working files; using these, the program can update embeddings when you add new interaction data. To refresh embeddings, point to new interaction files and run the program again (the output `npz` file will be overwritten).  

The `scripts/` folder can be used for scripts that prepare a dataset for use with the pipeline.

## Installation

1. Clone the repo and go to the project directory:

   ```bash
   git clone <repo-url>
   cd recsys
   ```

2. Install dependencies using **pip** or **conda**:

   **Option A — pip (venv)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # or:  .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

   **Option B — conda**

   ```bash
   conda env create -f requirements.yml
   conda activate recsys
   ```


## Usage

### 1. Configure paths and parameters

Edit `src/constants.py` and set:

- **`ITEM_EMBEDDINGS_FILE`** — path to the item embeddings `.npz` file (arrays: `item_id`, `embedding`).
- **`INTERACTIONS_FILES`** — list of Parquet paths (or a directory; all `*.parquet` inside are used). Each Parquet must have columns: `user_id`, `item_id`, `score`.
- **`OUTPUT_FILE`** — where to write the final user embeddings (default: `user_embeddings.npz`).
- **`WORK_DIR`** — directory for memmaps and checkpoints (default: `./data`).

You can also adjust `EMBED_DIM`, `BATCH_SIZE`, `SCORE_WEIGHTS`, `EMA_ALPHA`, `INITIAL_CAPACITY`, `MAX_CAPACITY`, `SAVE_EVERY_BATCHES`, and `GC_EVERY` in the same file.

### 2. Run the aggregator

From the project root:

```bash
python run.py
```

### 3. Output

- **Final embeddings:** `OUTPUT_FILE` (e.g. `user_embeddings.npz`) with:
  - `user_id` — 1D array of user IDs
  - `embedding` — 2D array of shape `(n_users, embed_dim)`, dtype `float16`

- **Working files** in `WORK_DIR`: memmaps, checkpoints, and user-id arrays (see `.gitignore`). Remove `WORK_DIR` to start from scratch.


## Requirements

- Python 3.8+
- See `requirements.txt` (pip) or `requirements.yml` (conda) for package versions.
