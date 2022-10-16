import os
import random
import typing
from dataclasses import dataclass
from pprint import pprint as print
from typing import List

import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
from dataclasses_json import dataclass_json
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.parsing.preprocessing import STOPWORDS
from gensim.test.utils import datapath
from gensim.utils import simple_preprocess
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.manifold import TSNE

nltk.download("abc")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")

SENTENCE_A = "Obama speaks to the media in Illinois"
SENTENCE_B = "The president greets the press in Chicago"

# typing for serialised model file
MODELSER_WORD2VEC = typing.TypeVar("model")
model_file = typing.NamedTuple("ModelFile", model=FlyteFile[MODELSER_WORD2VEC])

POS_TAG = "Noun"
# Set file names for train and test data
test_data_dir = os.path.join(gensim.__path__[0], "test", "test_data")
lee_train_file = os.path.join(test_data_dir, "lee_background.cor")
lee_test_file = os.path.join(test_data_dir, "lee.cor")

dataset = typing.NamedTuple(
    "GenerateSplitDataOutputs",
    train_data=typing.List[str],
    test_data=typing.List[str],
)


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, path):
        self.corpus_path = datapath(path)

    def __iter__(self):
        for line in open(self.corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield simple_preprocess(line)


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def build_train_test_dataset() -> List[List[str]]:
    # Set file names for train and test data
    sentences_train = MyCorpus(lee_train_file)
    train_corpus = list(sentences_train)
    print(train_corpus[:2])
    return train_corpus


# %%
# It is also possible in Flyte to pass custom objects, as long as they are
# declared as ``dataclass``es and also decorated with ``@dataclass_json``.
@dataclass_json
@dataclass
class Word2VecModelHyperparams(object):
    """
    These are the word-to-vec hyper parameters which can be set for training.
    """

    min_count: int = 1
    vector_size: int = 100
    workers: int = 4
    compute_loss: bool = True


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def train_model(
    training_data: List[List[str]], hyperparams: Word2VecModelHyperparams
) -> model_file:
    # instantiating and training the Word2Vec model
    model = gensim.models.Word2Vec(
        training_data,
        min_count=hyperparams.min_count,
        workers=hyperparams.workers,
        vector_size=hyperparams.vector_size,
        compute_loss=hyperparams.compute_loss,
    )
    training_loss = model.get_latest_training_loss()
    print(training_loss)
    file = serialise_model(model=model, file="word2vec.model")
    return file


def serialise_model(model: gensim.models.Word2Vec, file: str) -> model_file:
    model.save(file)
    return file


def deserialise_model(
    model_ser: FlyteFile[MODELSER_WORD2VEC],
) -> gensim.models.Word2Vec:
    model = gensim.models.Word2Vec.load(model_ser)
    return model


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def word_similarities(model_ser: FlyteFile[MODELSER_WORD2VEC]):
    model = deserialise_model(model_ser)
    wv = model.wv
    print(wv["nights"])
    print(wv.most_similar("nights"))


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def word_movers_distance(model_ser: FlyteFile[MODELSER_WORD2VEC]):
    sentences = [SENTENCE_A, SENTENCE_B]
    results = []
    for i in sentences:
        result = [w for w in utils.tokenize(i) if w not in STOPWORDS]
        results.append(result)
    model = deserialise_model(model_ser)
    distance = model.wv.wmdistance(*results)
    print(f"Word Movers Distance is {distance} (lower means closer)")


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def plot_with_matplotlib(x_vals, y_vals, labels):
    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)
    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))


@workflow
def nlp_workflow():
    split_data = build_train_test_dataset()
    train_model(training_data=split_data, hyperparams=Word2VecModelHyperparams())
    # word_similarities(model_ser=model.model)


# word_movers_distance(model=model.model)


if __name__ == "__main__":
    print(f"Running {__file__} main...")
    print(nlp_workflow())
