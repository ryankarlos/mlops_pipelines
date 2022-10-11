"""
TFRecord examples as in tensorflow docs:
https://www.tensorflow.org/tutorials/load_data/tfrecord#:~:text=The%20TFRecord%20format%20is%20a,to%20understand%20a%20message%20type.
"""
import os
from typing import Any, List, Tuple, Union

import flytekit
import numpy as np
import tensorflow as tf
from flytekit import Resources, task, workflow
from tensorflow.python.data.ops.dataset_ops import MapDataset

# The number of observations in the dataset.
n_observations = int(1e4)
# Boolean feature, encoded as False or True.
feature0 = np.random.choice([False, True], n_observations)
# Integer feature, random from 0 to 4.
feature1 = np.random.randint(0, 5, n_observations)
# String feature.
strings = np.array([b"cat", b"dog", b"chicken", b"horse", b"goat"])
feature2 = strings[feature1]
# Float feature, from a standard normal distribution.
feature3 = np.random.randn(n_observations)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feature0, feature1, feature2, feature3):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        "feature0": _int64_feature(feature0),
        "feature1": _int64_feature(feature1),
        "feature2": _bytes_feature(feature2),
        "feature3": _float_feature(feature3),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(f0, f1, f2, f3):
    tf_string = tf.py_function(
        serialize_example,
        (f0, f1, f2, f3),  # Pass these args to the above function.
        tf.string,
    )  # The return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar.


def decode_example(*args):
    serialized_example = serialize_example(*args)
    example_proto = tf.train.Example.FromString(serialized_example)
    return example_proto


@task(cache=True, cache_version="0.1", limits=Resources(mem="600Mi"))
def serialise_features_dataset(
    feature0: np.array,
    feature1: np.array,
    feature2: np.array,
    feature3: np.array,
) -> MapDataset:
    features_dataset = tf.data.Dataset.from_tensor_slices(
        (feature0, feature1, feature2, feature3)
    )
    serialized_features_dataset = features_dataset.map(tf_serialize_example)
    return serialized_features_dataset


@task(cache=True, cache_version="0.1", limits=Resources(mem="600Mi"))
def write_tfrecord(features_dataset: MapDataset, filename: str):
    # working_dir = flytekit.current_context().working_directory
    # fname = os.path.join(working_dir, filename)
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(features_dataset)


@task(cache=True, cache_version="0.1", limits=Resources(mem="600Mi"))
def read_tfrecord(filename: str):
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    for raw_record in raw_dataset.take(10):
        print(repr(raw_record))
    return raw_dataset


@workflow
def tf_record_wf(
    feature0: np.array,
    feature1: np.array,
    feature2: np.array,
    feature3: np.array,
    filename: str = "test.tfrecord",
):
    dataset = serialise_features_dataset(
        feature0=feature0, feature1=feature1, feature2=feature2, feature3=feature3
    )
    write_tfrecord(features_dataset=dataset, filename=filename)
    data_read = read_tfrecord(filename=filename)


if __name__ == "__main__":
    print(f"Running {__file__} main...")
    print(
        tf_record_wf(
            feature0=feature0, feature1=feature1, feature2=feature2, feature3=feature3
        )
    )
