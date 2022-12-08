import typing
import pandas as pd
import os

from flytekit import FlyteContext
from flytekit.models import literals
from flytekit.models.literals import StructuredDatasetMetadata
from flytekit.types.structured.structured_dataset import (
    PARQUET,
    StructuredDataset,
    StructuredDatasetDecoder,
    StructuredDatasetEncoder,
    StructuredDatasetTransformerEngine,
    StructuredDatasetType,
)


class IterablePandasToParquetEncodingHandler(StructuredDatasetEncoder):
    def __init__(self):
        super().__init__(pd.DataFrame, "/", PARQUET)

    def encode(
        self,
        ctx: FlyteContext,
        structured_dataset: StructuredDataset,
        structured_dataset_type: StructuredDatasetType,
    ) -> literals.StructuredDataset:

        path = typing.cast(str, structured_dataset.uri) or ctx.file_access.get_random_remote_directory()
        df = typing.cast(pd.DataFrame, structured_dataset.dataframe)
        local_dir = ctx.file_access.get_random_local_directory()
        partition_col = getattr(structured_dataset, "partition_col", None)
        local_path = local_dir if partition_col else os.path.join(local_dir, f"{0:05}")
        df.to_parquet(
            local_path,
            coerce_timestamps="us",
            allow_truncated_timestamps=False,
            partition_cols=partition_col,
        )
        print(f"Writing dataframe to folder {local_dir}")
        ctx.file_access.upload_directory(local_dir, path)
        structured_dataset_type.format = PARQUET
        return literals.StructuredDataset(uri=path, metadata=StructuredDatasetMetadata(structured_dataset_type))


class IterableParquetToPandasDecodingHandler(StructuredDatasetDecoder):
    def __init__(self):
        super().__init__(pd.DataFrame, "/", PARQUET)

    def decode(
        self,
        ctx: FlyteContext,
        flyte_value: literals.StructuredDataset,
        current_task_metadata: StructuredDatasetMetadata,
    ) -> typing.Union[pd.DataFrame, typing.Generator[pd.DataFrame, None, None]]:

        path = flyte_value.uri
        local_dir = ctx.file_access.get_random_local_directory()
        ctx.file_access.get_data(path, local_dir, is_multipart=True)
        partitioned_folders = os.listdir(flyte_value.uri)

        columns = None
        if current_task_metadata.structured_dataset_type and current_task_metadata.structured_dataset_type.columns:
            columns = [c.name for c in current_task_metadata.structured_dataset_type.columns]

        if len(partitioned_folders) == 1:
            return pd.read_parquet(local_dir, columns=columns)

        def read_partition(p):
            subfolder_src = os.path.join(flyte_value.uri, p)
            subfolder_dest = os.path.join(local_dir, p)
            print(f"Downloading {subfolder_src} to {subfolder_dest}")
            ctx.file_access.get_data(subfolder_src, subfolder_dest, is_multipart=True)
            return pd.read_parquet(subfolder_dest)

        return (read_partition(p) for p in partitioned_folders)


StructuredDatasetTransformerEngine.register(IterablePandasToParquetEncodingHandler(), override=True, default_for_type=True)
StructuredDatasetTransformerEngine.register(IterableParquetToPandasDecodingHandler(), override=True, default_for_type=True)
