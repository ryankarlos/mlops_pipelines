"""
Implementation of https://docs.flyte.org/projects/cookbook/en/latest/auto/core/type_system/structured_dataset.html
for showing how vaex df are accepted by the StructuredDataset
"""
from typing import Annotated

import numpy as np
import pandas as pd
import vaex
from flytekit import kwtypes, task, workflow
from flytekit.types.structured.structured_dataset import (
    PARQUET,
    StructuredDataset,
    StructuredDatasetTransformerEngine,
)

from flyte_transformers.vaex_typetransformer import (
    VaexDecodingHandlers,
    VaexEncodingHandlers,
)

superset_cols = kwtypes(Name=str, Age=int, Height=int)
subset_cols = kwtypes(Age=int)


StructuredDatasetTransformerEngine.register(
    VaexEncodingHandlers(vaex.DataFrame, supported_format=PARQUET)
)
StructuredDatasetTransformerEngine.register(
    VaexDecodingHandlers(vaex.DataFrame, supported_format=PARQUET)
)


@task
def to_vaex(
    ds: Annotated[StructuredDataset, subset_cols]
) -> Annotated[StructuredDataset, subset_cols, PARQUET]:
    vaex_df = ds.open(np.ndarray).all()
    return StructuredDataset(dataframe=vaex_df)


@task
def get_df(a: int) -> Annotated[vaex.DataFrame, superset_cols]:
    """
    Generate a sample dataframe
    """

    df = pd.DataFrame({"Name": ["Tom", "Joseph"], "Age": [a, 22], "Height": [160, 178]})
    vaex_df = vaex.from_pandas(df)
    return vaex_df


@task
def get_subset_df(
    df: Annotated[vaex.DataFrame, subset_cols]
) -> Annotated[StructuredDataset, subset_cols]:
    df = df.open(vaex.DataFrame).all()
    df = vaex.concat([df, vaex.from_pandas_df(pd.DataFrame([[30]], columns=["Age"]))])
    return StructuredDataset(dataframe=df)


@workflow
def vaex_compatibility_wf(a: int) -> Annotated[StructuredDataset, subset_cols]:
    df = get_df(a=a)
    ds = get_subset_df(df=df)  # noqa
    return to_vaex(ds=ds)


if __name__ == "__main__":
    vaex_df_one = vaex_compatibility_wf(a=42).open(vaex.DataFrame).all()
    print(f"vaex DataFrame compatibility check output: {vaex_df_one}")
