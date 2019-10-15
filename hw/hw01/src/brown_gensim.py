import argparse
from pathlib import Path
from typing import List

import gensim
import nltk
from nltk.corpus import brown
from nltk.data import find

MODEL_PATH = Path("model/brown.embedding")


def parse_args():
    r""" Parse the command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("k", help="Top-K words to return", type=int)
    parser.add_argument("--words", help="Word to test", type=str, nargs="+")
    args = parser.parse_args()

    if args.k <= 0: raise ValueError("k must be positive")
    return args


def _get_brown_model():
    r""" Gets the Brown Word2Vec model """
    if MODEL_PATH.exists():
        model = gensim.models.Word2Vec.load(str(MODEL_PATH))
    else:
        nltk.download('brown')
        model = gensim.models.Word2Vec(brown.sents())
        MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
        model.save(str(MODEL_PATH))
    return model


def _get_google_news_model():
    word2vec_path = "./model/GoogleNews-vectors-negative300.bin"
    return gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


def _get_top_k(mod_name: str, model, words: List[str], k: int):
    r""" Get the top \p k most similar words (by cosine similarity) for each word in \p words """
    for w in words:
        print(f"{mod_name}: Word to Find Most Similar: {w}")
        print(model.most_similar(positive=[w], topn=k))
        print("")


def _main(words: List[str], k: int):
    model = _get_brown_model()
    _get_top_k("Brown", model.wv, words, k)

    model = _get_google_news_model()
    _get_top_k("GoogleNews", model, words, k)


if __name__ == '__main__':
    _args = parse_args()
    _main(words=_args.words, k=_args.k)
