import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from tqdm import tqdm

INPUT_FILE = Path("example/path/to/filtered.parquet")
OUTPUT_DIR = INPUT_FILE.parent / "splitted_small"

NUM_PARTS = 5
FRACTION = 1 / 3
BATCH_SIZE = 1_000_000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def split_file():
    parquet_file = pq.ParquetFile(INPUT_FILE)

    total_rows = parquet_file.metadata.num_rows
    target_rows = int(total_rows * FRACTION)

    rows_per_part = target_rows // NUM_PARTS

   

    writers = []
    for i in range(NUM_PARTS):
        out_path = OUTPUT_DIR / f"{INPUT_FILE.stem}_part{i+1}.parquet"
        writers.append((out_path, None, 0))

    current_part = 0
    written_total = 0

    with tqdm(total=target_rows, unit="rows") as pbar:
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):

            if written_total >= target_rows:
                break

            batch_rows = batch.num_rows

            if written_total + batch_rows > target_rows:
                batch = batch.slice(0, target_rows - written_total)
                batch_rows = batch.num_rows

            offset = 0
            while offset < batch_rows and current_part < NUM_PARTS:

                out_path, writer, written_in_part = writers[current_part]

                remaining_in_part = rows_per_part - written_in_part
                take = min(remaining_in_part, batch_rows - offset)

                sub_batch = batch.slice(offset, take)

                if writer is None:
                    writer = pq.ParquetWriter(
                        out_path,
                        sub_batch.schema,
                        compression="zstd",
                        use_dictionary=True
                    )
                    writers[current_part] = (out_path, writer, written_in_part)

                writer.write_batch(sub_batch)

                written_in_part += take
                writers[current_part] = (out_path, writer, written_in_part)

                offset += take
                written_total += take
                pbar.update(take)

                if written_in_part >= rows_per_part:
                    writer.close()
                    current_part += 1

    for out_path, writer, _ in writers:
        if writer is not None:
            writer.close()

  


if __name__ == "__main__":
    split_file()
