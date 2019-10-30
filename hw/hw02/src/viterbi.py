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
    args.add_argument("--smooth", help="Perform add-1 smoothing", action="store_true")
    args = args.parse_args()

    for fld in ["train", "test"]:
        path = Path(args.__getattribute__(fld))
        if not path.is_file(): raise ValueError(f"Unknown {fld} file {path}")
        args.__setattr__(fld, path)
    args.out = Path(args.out)
    if args.out.exists() and not args.out.is_file():
        raise ValueError(f"Out file \"{args.out}\" appears not to be a file.  Cannot overwrite")
    return args


def perform_viterbi(sentence: UnlabeledSentence, prob_struct: ProbStruct) -> List[str]:
    vit_dp = np.zeros((prob_struct.num_state(), len(sentence) + 2))
    best_prev = np.zeros(vit_dp.shape, dtype=np.int32)

    # Set first column for start symbols
    vit_dp[prob_struct.get_pos_id(prob_struct.START), 0] = 1
    # Inner columns are for sentence words
    for col, word in zip(range(1, len(sentence) + 2), sentence):  # offset range for START
        prev_prob = vit_dp[:, col - 1]
        likelihood_vec = prob_struct.get_likelihood_vec(word)
        for row in range(1, prob_struct.num_state() - 1):  # ignore end
            p_vec = prob_struct.get_transition_prob_vec(row)
            combo = prev_prob * likelihood_vec[row] * p_vec

            vit_dp[row, col] = np.max(combo)
            best_prev[row, col] = np.argmax(combo)
    # Get final probability and last tag
    end_pos_id = prob_struct.get_pos_id(prob_struct.END)
    combo = vit_dp[:, -2] * prob_struct.get_transition_prob_vec(end_pos_id)
    vit_dp[end_pos_id, -1] = np.max(combo)
    best_prev[end_pos_id, -1] = np.argmax(combo)

    # Reconstruct sequence of labels
    lbls = [best_prev[end_pos_id, -1]]
    # Start before last element which was manually added above
    for j in range(best_prev.shape[1] - 2, 1, -1):
        lbls.append(best_prev[lbls[-1], j])
    return [prob_struct.lookup_pos(pos_id) for pos_id in lbls[::-1]]


def _main(args: Namespace):
    train_corpus = import_labeled_corpus(args.train)
    prob_struct = ProbStruct(train_corpus, smooth=args.smooth)
    test_corpus = import_test_corpus(args.test)

    seq_lbls = []
    for sentence in test_corpus:
        seq_lbls.append(perform_viterbi(sentence, prob_struct))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(args.out), "w+") as f_out:
        for i, (sentence, lbls) in enumerate(zip(test_corpus, seq_lbls)):
            if i > 0: f_out.write("\n")
            for word, lbl in zip(sentence, lbls):
                f_out.write(f"{word}\t{lbl}\n")
        f_out.write("\n")  # File has an extra space I assume to make extra clear last is sentence


if __name__ == '__main__':
    _main(parse_args())
