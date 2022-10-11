import os
import typing
from typing import Type

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import vaex
from flytekit import FlyteContext, StructuredDatasetType
from flytekit.core.type_engine import T, TypeEngine, TypeTransformer
from flytekit.models import literals
from flytekit.models.literals import Literal, Scalar, Schema, StructuredDatasetMetadata
from flytekit.models.types import LiteralType, SchemaType
from flytekit.types.schema import (
    SchemaEngine,
    SchemaFormat,
    SchemaHandler,
    SchemaReader,
    SchemaWriter,
)
from flytekit.types.structured.structured_dataset import (
    PARQUET,
    StructuredDataset,
    StructuredDatasetDecoder,
    StructuredDatasetEncoder,
    StructuredDatasetTransformerEngine,
)

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated


class VaexEncodingHandlers(StructuredDatasetEncoder):
    def encode(
        self,
        ctx: FlyteContext,
        structured_dataset: StructuredDataset,
        structured_dataset_type: StructuredDatasetType,
    ) -> literals.StructuredDataset:
        df = typing.cast(vaex.DataFrame, structured_dataset.dataframe)
        name = ["col" + str(i) for i in range(len(df))]
        table = pa.Table.from_arrays(df.to_pandas_df(), name)
        path = ctx.file_access.get_random_remote_directory()
        local_dir = ctx.file_access.get_random_local_directory()
        local_path = os.path.join(local_dir, f"{0:05}")
        pq.write_table(table, local_path)
        ctx.file_access.upload_directory(local_dir, path)
        return literals.StructuredDataset(
            uri=path,
            metadata=StructuredDatasetMetadata(
                structured_dataset_type=StructuredDatasetType(format=PARQUET)
            ),
        )


class VaexDecodingHandlers(StructuredDatasetDecoder):
    def decode(
        self,
        ctx: FlyteContext,
        flyte_value: literals.StructuredDataset,
        current_task_metadata: StructuredDatasetMetadata,
    ) -> np.ndarray:
        local_dir = ctx.file_access.get_random_local_directory()
        ctx.file_access.get_data(flyte_value.uri, local_dir, is_multipart=True)
        table = pq.read_table(local_dir)
        return vaex.from_arrow_table(table)


class VaexDataFrameSchemaReader(SchemaReader[vaex.DataFrame]):
    """
    Implements how VaexDataFrame should be read using the ``open`` method of FlyteSchema
    """

    def __init__(
        self,
        local_dir: os.PathLike,
        cols: typing.Optional[typing.Dict[str, type]],
        fmt: SchemaFormat,
    ):
        super().__init__(local_dir, cols, fmt)

    def _read(self, path: os.PathLike, **kwargs) -> vaex.DataFrame:
        return vaex.open(path, **kwargs)


class VaexDataFrameSchemaWriter(SchemaWriter[vaex.DataFrame]):
    """
    Implements how VaexDataFrame should be written to using ``open`` method of FlyteSchema
    """

    def __init__(
        self,
        local_dir: os.PathLike,
        cols: typing.Optional[typing.Dict[str, type]],
        fmt: SchemaFormat,
    ):
        super().__init__(local_dir, cols, fmt)

    def _write(self, df: T, path: os.PathLike, **kwargs):
        return df.export_parquet(
            path,
            **kwargs,
        )


class VaexDataFrameTransformer(TypeTransformer[vaex.DataFrame]):
    """
    Transforms a vaex DataFrame to Schema without column types.
    """

    def __init__(self):
        super().__init__("VaexDataFrame<->GenericSchema", vaex.DataFrame)

    @staticmethod
    def _get_schema_type() -> SchemaType:
        return SchemaType(columns=[])

    def get_literal_type(self, t: Type[vaex.DataFrame]) -> LiteralType:
        return LiteralType(schema=self._get_schema_type())

    def to_literal(
        self,
        ctx: FlyteContext,
        python_val: vaex.DataFrame,
        python_type: Type[vaex.DataFrame],
        expected: LiteralType,
    ) -> Literal:
        local_dir = ctx.file_access.get_random_local_directory()
        w = VaexDataFrameSchemaWriter(
            local_dir=local_dir, cols=None, fmt=SchemaFormat.PARQUET
        )
        w.write(python_val)
        remote_path = ctx.file_access.get_random_remote_directory()
        ctx.file_access.put_data(local_dir, remote_path, is_multipart=True)
        return Literal(
            scalar=Scalar(schema=Schema(remote_path, self._get_schema_type()))
        )

    def to_python_value(
        self, ctx: FlyteContext, lv: Literal, expected_python_type: Type[vaex.DataFrame]
    ) -> vaex.DataFrame:
        if not (lv and lv.scalar and lv.scalar.schema):
            return vaex.DataFrame()
        local_dir = ctx.file_access.get_random_local_directory()
        ctx.file_access.get_data(lv.scalar.schema.uri, local_dir, is_multipart=True)
        r = VaexDataFrameSchemaReader(
            local_dir=local_dir, cols=None, fmt=SchemaFormat.PARQUET
        )
        return r.all()

    def to_html(
        self,
        ctx: FlyteContext,
        python_val: vaex.DataFrame,
        expected_python_type: Type[T],
    ):
        return python_val.describe().to_html()


SchemaEngine.register_handler(
    SchemaHandler(
        "vaex-dataframe-schema",
        vaex.DataFrame,
        VaexDataFrameSchemaReader,
        VaexDataFrameSchemaWriter,
    )
)
TypeEngine.register(VaexDataFrameTransformer())
