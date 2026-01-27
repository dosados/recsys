import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

BATCH_SIZE = 1_000_000

INPUT_FILES = [
    "example/path/to/raw_interactions.parquet",
]

NUM_PROCESSES = 3


def process_file(input_file):
    input_file = Path(input_file)
    output_file = input_file.with_name(input_file.stem + "_filtered_file.parquet")

    print(f"\nProcessing file: {input_file.name}")

    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows

    writer = None

    with tqdm(total=total_rows, unit="rows") as pbar:
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
            like = batch.column(batch.schema.get_field_index("like")).to_numpy(zero_copy_only=False)
            share = batch.column(batch.schema.get_field_index("share")).to_numpy(zero_copy_only=False)
            bookmark = batch.column(batch.schema.get_field_index("bookmark")).to_numpy(zero_copy_only=False)
            click = batch.column(batch.schema.get_field_index("click_on_author")).to_numpy(zero_copy_only=False)
            comments = batch.column(batch.schema.get_field_index("open_comments")).to_numpy(zero_copy_only=False)
            timespent = batch.column(batch.schema.get_field_index("timespent")).to_numpy(zero_copy_only=False)

            user_id = batch.column(batch.schema.get_field_index("user_id")).to_numpy(zero_copy_only=False)
            item_id = batch.column(batch.schema.get_field_index("item_id")).to_numpy(zero_copy_only=False)

            interaction_sum_full = (
                like.astype(np.int8)
                + share.astype(np.int8)
                + bookmark.astype(np.int8)
                + click.astype(np.int8)
                + comments.astype(np.int8)
            )


            informative_mask = ~((interaction_sum_full < 1) & (timespent < 15))

            if not informative_mask.any():
                pbar.update(batch.num_rows)
                continue

            user_id = user_id[informative_mask]
            item_id = item_id[informative_mask]
            interaction_sum = interaction_sum_full[informative_mask]

            score = np.zeros(len(interaction_sum), dtype=np.int8)

            score[interaction_sum >= 4] = 3
            mask2 = (interaction_sum >= 2) & (score == 0)
            score[mask2] = 2
            mask1 = (interaction_sum >= 1) & (score == 0)
            score[mask1] = 1

            arrays = [
                pa.array(user_id, type=pa.uint32()),
                pa.array(item_id, type=pa.uint32()),
                pa.array(score, type=pa.int8()),
            ]

            schema = pa.schema([
                ("user_id", pa.uint32()),
                ("item_id", pa.uint32()),
                ("score", pa.int8()),
            ])

            new_batch = pa.RecordBatch.from_arrays(arrays, schema=schema)

            if writer is None:
                writer = pq.ParquetWriter(
                    output_file,
                    schema,
                    compression="zstd",
                    use_dictionary=True
                )

            writer.write_batch(new_batch)
            pbar.update(batch.num_rows)

    if writer is not None:
        writer.close()

    print(f"Done: {output_file}")




if __name__ == "__main__":
    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(process_file, INPUT_FILES)
