EMBED_DIM = 32
BATCH_SIZE = 250_000


SCORE_WEIGHTS = {
    0: -0.05,
    1: 1.0,
    2: 2.0,
    3: 3.0
}


INITIAL_CAPACITY = 4_000_000  # Initial capacity for user embeddings
MAX_CAPACITY = 50_000_000     # Maximum capacity for user embeddings


SAVE_EVERY_BATCHES = 30  # Save checkpoint every N batches
GC_EVERY = 4            # Run garbage collection every N batches


EMA_ALPHA = 0.95  # p in the formula: new_emb = p * old_emb + (1-p) * batch_emb


ITEM_EMBEDDINGS_FILE = "example/path/to/item_embeddings.npz"

INTERACTIONS_FILES = [
    "example/path/to/interactions.parquet",
]

OUTPUT_FILE = "user_embeddings.npz"
WORK_DIR = "./data"
