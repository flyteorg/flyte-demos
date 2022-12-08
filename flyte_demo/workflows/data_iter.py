"""An example of custom StructuredDataset encoder/decoder."""

import random
import string
from flytekit import task, workflow

import pandas as pd

import custom_types

from flytekit.types.structured.structured_dataset import (
    StructuredDataset,
)


@task
def make_df() -> StructuredDataset:
    df = pd.DataFrame.from_records([
        {
            "id": i,
            "partition": (i % 10) + 1,
            "name": "".join(
                random.choices(string.ascii_uppercase + string.digits, k=10)
            )
        }
        for i in range(1000)
    ])
    sd = StructuredDataset(dataframe=df)
    sd.partition_col = "partition"
    return sd


@task
def use_df(dataset: StructuredDataset) -> pd.DataFrame:
    output = []
    for dd in dataset.open(pd.DataFrame).iter():
        print(f"This is a partial dataframe")
        print(dd.head(3))
        output.append(dd)
    return pd.concat(output)


@workflow
def wf() -> pd.DataFrame:
    df = make_df()
    return use_df(dataset=df)


if __name__ == "__main__":
    print(f"dataset: {wf()}")
