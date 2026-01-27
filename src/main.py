import logging
import sys

from .aggregator import UserEmbeddingAggregator
from .constants import (
    EMBED_DIM,
    BATCH_SIZE,
    SCORE_WEIGHTS,
    EMA_ALPHA,
    INITIAL_CAPACITY,
    MAX_CAPACITY,
    SAVE_EVERY_BATCHES,
    GC_EVERY,
    ITEM_EMBEDDINGS_FILE,
    INTERACTIONS_FILES,
    OUTPUT_FILE,
    WORK_DIR
)



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""

    logger.info("Using default constants (edit constants.py to change)")
    item_embeddings_file = ITEM_EMBEDDINGS_FILE
    interactions_files = INTERACTIONS_FILES
    output_file = OUTPUT_FILE
    work_dir = WORK_DIR
    embed_dim = EMBED_DIM
    batch_size = BATCH_SIZE
    score_weights = SCORE_WEIGHTS
    ema_alpha = EMA_ALPHA
    initial_capacity = INITIAL_CAPACITY
    max_capacity = MAX_CAPACITY
    save_every_batches = SAVE_EVERY_BATCHES
    gc_every = GC_EVERY
    
    aggregator = UserEmbeddingAggregator(
        item_embeddings_file=item_embeddings_file,
        interactions_files=interactions_files,
        output_file=output_file,
        work_dir=work_dir,
        embed_dim=embed_dim,
        batch_size=batch_size,
        score_weights=score_weights,
        ema_alpha=ema_alpha,
        initial_capacity=initial_capacity,
        max_capacity=max_capacity,
        save_every_batches=save_every_batches,
        gc_every=gc_every
    )
    
    try:
        aggregator.process()
        aggregator.finalize()
        logger.info("Successfully completed!")
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Checkpoint saved. Resume by running again.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
