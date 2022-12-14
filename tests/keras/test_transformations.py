from collections import OrderedDict

import flytekit
import keras
import numpy as np
import pytest
from flytekit import task
from flytekit.configuration import Image, ImageConfig
from flytekit.core import context_manager
from flytekit.extras.keras import KerasModelTransformer, KerasSequentialTransformer
from flytekit.models.core.types import BlobType
from flytekit.models.literals import BlobMetadata
from flytekit.models.types import LiteralType
from flytekit.tools.translator import get_serializable

default_img = Image(name="default", fqn="test", tag="tag")
serialization_settings = flytekit.configuration.SerializationSettings(
    project="project",
    domain="domain",
    version="version",
    env=None,
    image_config=ImageConfig(default_image=default_img, images=[default_img]),
)


def build_keras_sequential_model():
    model = keras.Sequential()
    model.add(keras.Input(shape=(16,)))
    model.add(keras.layers.Dense(8))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer="sgd", loss="mse")
    return model


def build_keras_model_class() -> keras.Model:
    inputs = keras.Input(shape=(16,))
    x = keras.layers.Dense(8)(inputs)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="mse")
    return model


@pytest.mark.parametrize(
    "transformer,python_type,format",
    [
        (
            KerasSequentialTransformer(),
            keras.Sequential,
            KerasSequentialTransformer.KERAS_FORMAT,
        ),
        (KerasModelTransformer(), keras.Model, KerasModelTransformer.KERAS_FORMAT),
    ],
)
def test_get_literal_type(transformer, python_type, format):
    tf = transformer
    lt = tf.get_literal_type(python_type)
    assert lt == LiteralType(
        blob=BlobType(format=format, dimensionality=BlobType.BlobDimensionality.SINGLE)
    )


@pytest.mark.parametrize(
    "transformer,python_type,format,python_val",
    [
        (
            KerasSequentialTransformer(),
            keras.Sequential,
            KerasSequentialTransformer.KERAS_FORMAT,
            build_keras_sequential_model(),
        ),
        (
            KerasModelTransformer(),
            keras.Model,
            KerasModelTransformer.KERAS_FORMAT,
            build_keras_model_class(),
        ),
    ],
)
def test_to_python_value_and_literal(transformer, python_type, format, python_val):
    ctx = context_manager.FlyteContext.current_context()
    tf = transformer
    lt = tf.get_literal_type(python_type)
    lv = tf.to_literal(ctx, python_val, type(python_val), lt)  # type: ignore
    assert lv.scalar.blob.metadata == BlobMetadata(
        type=BlobType(
            format=format,
            dimensionality=BlobType.BlobDimensionality.SINGLE,
        )
    )
    assert lv.scalar.blob.uri is not None

    output = tf.to_python_value(ctx, lv, python_type)
    if isinstance(python_val, (keras.Sequential, keras.Model)):
        for p1, p2 in zip(output.weights, python_val.weights):
            np.testing.assert_array_equal(p1.numpy(), p2.numpy())
        assert True
    else:
        assert isinstance(output, dict)


def test_example_model():
    @task
    def t1() -> keras.Sequential:
        return build_keras_sequential_model()

    @task
    def t2() -> keras.Model:
        return build_keras_model_class()

    task_spec1 = get_serializable(OrderedDict(), serialization_settings, t1)
    task_spec2 = get_serializable(OrderedDict(), serialization_settings, t2)
    assert (
        task_spec1.template.interface.outputs["o0"].type.blob.format
        is KerasSequentialTransformer.KERAS_FORMAT
    )
    assert (
        task_spec2.template.interface.outputs["o0"].type.blob.format
        is KerasModelTransformer.KERAS_FORMAT
    )
