from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import numpy as np

from _importer import import_labeled_corpus
from _prob_struct import ProbStruct, UnlabeledSentence


def parse_args():
    r""" Parse the command line arguments """
    args = ArgumentParser()
    args.add_argument("train", help="Path to the training file")
    args.add_argument("test", help="Path to the test file")
    args = args.parse_args()

    for fld in ["train", "test"]:
        path = Path(args.__getattr__(fld))
        if not path.is_file(): raise ValueError(f"Unknown {fld} file {path}")
        args.__setattr__(fld, path)
    return args


def perform_viterbi(sentence: UnlabeledSentence, probs: ProbStruct) -> List[str]:
    probs = np.zeros((probs.num_state(), len(sentence) + 2))
    probs[0, probs.(probs.START)] = 1
    for i, word in enumerate(sentence):



def _main(args: Namespace):
    train_corpus = import_labeled_corpus(args.train)

    test_corpus = import

if __name__ == '__main__':
    _main(parse_args())
