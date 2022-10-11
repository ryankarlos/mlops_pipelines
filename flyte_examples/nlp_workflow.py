import typing
import unicodedata
from typing import List

import nltk
import numpy as np
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from gensim.models.doc2vec import Word2Vec
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download("abc")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")


text = (
    "The elephant sneezed at the sight of potatoes. Bats can see via echolocation. "
    "See the bat sight sneeze!.Wondering, she opened the door to the studio."
)

# working_dir = flytekit.current_context().working_directory
# flytefile = os.path.join(working_dir, "doc2vec.model")
MODELSER_DOC2VEC = typing.TypeVar("model")
model_file = typing.NamedTuple("ModelFile", model=FlyteFile[MODELSER_DOC2VEC])
workflow_outputs = typing.NamedTuple(
    "WorkflowOutputs", model=FlyteFile[MODELSER_DOC2VEC], vector=np.array
)


def is_punct(token):
    return all(unicodedata.category(char).startswith("P") for char in token)


def is_stopword(token):
    return token.lower() in set(nltk.corpus.stopwords.words("english"))


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def tokenize(text: str) -> List[str]:
    text = text.lower()
    return nltk.word_tokenize(text)


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def stemmer(corpus: List[str]) -> List[str]:
    stem = nltk.stem.SnowballStemmer("english")
    return [
        stem.stem(token)
        for token in corpus
        if not is_punct(token) and not is_stopword(token)
    ]


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def lemmatize(corpus: List[str], pos_tag: str) -> List[str]:
    tag = {
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
        "J": wordnet.ADJ,
    }.get(pos_tag[0], wordnet.NOUN)

    return [WordNetLemmatizer().lemmatize(token, tag) for token in corpus]


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def train_doc2vec(corpus: List[str], file: MODELSER_DOC2VEC) -> model_file:
    model = Word2Vec(corpus, vector_size=5, min_count=0)
    model.save(file)
    return file


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def word_vector(model: model_file.model) -> np.array:
    return model.wv.vectors


@workflow
def nltk_workflow(text: str) -> workflow_outputs:
    """ """
    corpus = tokenize(text=text)
    stemmed = stemmer(corpus=corpus)
    lemma = lemmatize(corpus=stemmed, pos_tag="Noun")
    model = train_doc2vec(corpus=lemma, file="doc2vec.model")
    vec = word_vector(model=model.model)
    return model.model, vec


if __name__ == "__main__":
    print(f"Running {__file__} main...")
    print(nltk_workflow(text=text))
