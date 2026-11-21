"""Read the first rows of a Hub dataset without downloading the whole split.

``load_dataset(..., streaming=True)`` leaves Arrow I/O threads running when the
iteration stops early, and those threads crash the interpreter at shutdown
(``PyGILState_Release``). Reading the Parquet shards on the calling thread and
closing each file explicitly avoids that.
"""

from collections.abc import Iterator
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset_builder


def iter_head_rows(
    path: str, name: str | None = None, *, split: str, batch_size: int = 8
) -> Iterator[dict[str, Any]]:
    """Yield decoded rows of ``split`` in order, reading shard by shard."""
    builder = load_dataset_builder(path, name)
    features = builder.info.features
    for url in builder.config.data_files[split]:
        with fsspec.open(url, "rb") as f:
            parquet_file = pq.ParquetFile(f)
            try:
                for batch in parquet_file.iter_batches(
                    batch_size=batch_size, use_threads=False
                ):
                    table = pa.Table.from_batches([batch])
                    ds = Dataset(table)
                    if features is not None:
                        ds = ds.cast(features)
                    yield from ds
            finally:
                parquet_file.close()
