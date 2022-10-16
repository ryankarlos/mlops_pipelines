"""
Implementation of https://docs.flyte.org/projects/cookbook/en/latest/auto/core/type_system/structured_dataset.html
for showing how flytekit-vaex df are accepted by the StructuredDataset
"""
from typing import Annotated

import pandas as pd
import vaex
from flytekit import kwtypes, task, workflow
from flytekit.types.structured.structured_dataset import (
    PARQUET,
    StructuredDataset,
    StructuredDatasetTransformerEngine,
)
from flytekitplugins.vaex.sd_transformer import (
    ParquetToVaxDataFrameDecodingHandler,
    VaexDataFrameRenderer,
    VaexDataFrameToParquetEncodingHandlers,
)

superset_cols = kwtypes(Name=str, Age=int, Height=int)
subset_cols = kwtypes(Age=int)


StructuredDatasetTransformerEngine.register(VaexDataFrameToParquetEncodingHandlers())
StructuredDatasetTransformerEngine.register(ParquetToVaxDataFrameDecodingHandler())
StructuredDatasetTransformerEngine.register_renderer(
    vaex.DataFrame, VaexDataFrameRenderer()
)


subset_schema = Annotated[StructuredDataset, kwtypes(col2=str), PARQUET]


@task
def generate() -> subset_schema:
    pd_df = pd.DataFrame({"col1": [1, 3, 2], "col2": list("abc")})
    vaex_df = vaex.from_pandas(pd_df)
    return StructuredDataset(dataframe=vaex_df)


@task
def consume(df: subset_schema) -> subset_schema:
    df = df.open(vaex.DataFrame).all()

    assert df["col2"][0] == "a"
    assert df["col2"][1] == "b"
    assert df["col2"][2] == "c"

    return StructuredDataset(dataframe=df)


@task
def vaex_renderer(df: subset_schema):
    df = df.open(vaex.DataFrame).all()
    assert VaexDataFrameRenderer().to_html(df) == pd.DataFrame(
        df.describe().transpose(), columns=df.describe().columns
    ).to_html(index=False)


@workflow
def wf():
    df = generate()
    df = consume(df=df)
    vaex_renderer(df=df)


if __name__ == "__main__":
    wf()
    # assert result is not None
