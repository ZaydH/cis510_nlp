import argparse
from argparse import Namespace

from load_newsgroups import load_20newsgroups
from pubn.model import NlpBiasedLearner
from pubn.pu_loss import LossType


def parse_args() -> Namespace:
    args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument("size_p", "Number of elements in the POSITIVE (labeled) set", type=int)
    args.add_argument("size_n", "Number of elements in the biased NEGATIVE (labeled) set", type=int)
    args.add_argument("loss", choices=[e.name for e in LossType])
    args.add_argument("--bs", help="Batch size", type=int)
    args.add_argument("--tau", help="Hyperparameter used to determine eta", type=float)
    args.add_argument("--pos", nargs='+', type=int)
    args.add_argument("--neg", nargs='+', type=int)

    args = args.parse_args()
    # Arguments error checking
    if args.size_p <= 0: raise ValueError("size_p must be positive valued")
    if args.bs <= 0: raise ValueError("bs must be positive valued")

    # Configure any related fields
    args.loss = LossType[args.loss]

    args.pos, args.neg = set(args.pos), set(args.neg)
    assert not (args.pos & args.neg), "Positive and negative classes not disjoint"

    return args


def _main(args: Namespace):
    TEXT, LABEL, train_ds, test_ds = load_20newsgroups(args)
    learner = NlpBiasedLearner(args, TEXT.vocab.vectors)


if __name__ == "__main__":
    _main(parse_args())
