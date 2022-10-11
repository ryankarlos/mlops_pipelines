"""
Implements the example in https://www.tensorflow.org/tutorials/quickstart/advanced
"""

import os
from typing import Any, List, NamedTuple, Tuple

import flytekit
import joblib
import numpy as np
import tensorflow as tf
from flytekit import Resources, task, workflow
from flytekit.types.file import JoblibSerializedFile
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten

EPOCHS = 2
BATCH_SIZE = 64

dataset = NamedTuple(
    "GenerateSplitDataOutputs",
    x_train=np.array,
    y_train=np.array,
    x_test=np.array,
    y_test=np.array,
)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


@task(cache=True, cache_version="0.1", limits=Resources(mem="600Mi"))
def generate_and_split_data() -> dataset:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")
    return (x_train, y_train, x_test, y_test)


@task(cache_version="1.0", cache=True, limits=Resources(mem="600Mi"))
def fit(x: np.array, y: np.array, epochs: int, batch_size: int) -> JoblibSerializedFile:
    model = MyModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    # fetch the features and target columns from the train dataset
    # train_loss = tf.keras.metrics.Mean(name="train_loss")
    model.compile(
        optimizer=optimizer,
        loss=loss_object,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.fit(x, y, epochs=epochs)
    working_dir = flytekit.current_context().working_directory
    fname = os.path.join(working_dir, f"model.joblib.dat")
    joblib.dump(model, fname)

    # return the serialized model
    return JoblibSerializedFile(path=fname)


@task(cache_version="1.0", cache=True, limits=Resources(mem="600Mi"))
def evaluate_metrics(
    x: np.array,
    y: np.array,
    model_ser: JoblibSerializedFile,
):
    # load the model
    model = joblib.load(model_ser)
    results = model.evaluate(x, y, verbose=2)
    return results


@task(cache_version="1.0", cache=True, limits=Resources(mem="600Mi"))
def predict(model_ser: JoblibSerializedFile, test_data: np.array) -> Any:
    # load the model
    model = joblib.load(model_ser)
    print("Generate predictions for 3 samples")
    predictions = model.predict(test_data[:3])
    print("predictions shape:", predictions.shape)
    return predictions


@workflow
def tensorflow_mnist_workflow(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):

    # generate the data and split it into train test, and validation data
    split_data_vals = generate_and_split_data()

    # fit the XGBoost model
    model = fit(
        x=split_data_vals.x_train,
        y=split_data_vals.y_train,
        epochs=epochs,
        batch_size=batch_size,
    )

    # generate predictions
    eval_results = evaluate_metrics(
        model_ser=model, x=split_data_vals.x_test, y=split_data_vals.y_test
    )
    predictions = predict(model_ser=model, test_data=split_data_vals.x_test)


if __name__ == "__main__":
    print(f"Running {__file__} main...")
    print("TensorFlow version:", tf.__version__)
    print(tensorflow_mnist_workflow())
