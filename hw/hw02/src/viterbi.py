from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import numpy as np

from _importer import import_labeled_corpus, import_test_corpus
from _prob_struct import ProbStruct, UnlabeledSentence


def parse_args():
    r""" Parse the command line arguments """
    args = ArgumentParser()
    args.add_argument("train", help="Path from where to read the training file")
    args.add_argument("test", help="Path from where to read the test file")
    args.add_argument("out", help="Path write the generated labels")
    args.add_argument("smooth", help="Perform add-1 smoothing", action="store_true")
    args = args.parse_args()

    for fld in ["train", "test"]:
        path = Path(args.__getattribute__(fld))
        if not path.is_file(): raise ValueError(f"Unknown {fld} file {path}")
        args.__setattr__(fld, path)
    return args


def perform_viterbi(sentence: UnlabeledSentence, probs: ProbStruct) -> List[str]:
    probs = np.zeros((probs.num_state(), len(sentence) + 2))
    best_prev = probs.copy()

    # Set initial setup
    probs[0, probs.get_pos_id(probs.START)] = 1
    best_prev[:, 1] = probs.get_pos_id(probs.START)

    for col in range(1, len(sentence) + 2):  # Add to since start at 1 and need to consider END
        prev_prob = best_prev[:, col - 1]
        p_vec = probs.get_transition_prob_vec(col)
        for row in range(1, len(sentence) + 2):
            likelihood_vec = probs.get_likelihood_vec(sentence[col - 1])
            combo = prev_prob * likelihood_vec * p_vec

            probs[row, col] = np.max(combo)
            best_prev[row, col + 1] = np.argmax(combo)

    lbls = [best_prev.shape[1][-1]]
    for j in range(best_prev.shape[1][-1] - 1, 1, step=-1):  # Start after last
        lbls.append(best_prev[lbls[-1], j])
    return [probs.lookup_pos(pos_id) for pos_id in lbls[::-1]]


def _main(args: Namespace):
    train_corpus = import_labeled_corpus(args.train)
    prob_struct = ProbStruct(train_corpus, smooth=args.smooth)
    test_corpus = import_test_corpus(args.test)

    seq_lbls = []
    for sentence in test_corpus:
        seq_lbls.append(perform_viterbi(sentence, prob_struct))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(args.out), "w+") as f_out:
        for i, sentence in enumerate(seq_lbls):
            if i > 0: f_out.write("\n")
            f_out.write("\t".join(sentence))


if __name__ == '__main__':
    _main(parse_args())
