import argparse
from argparse import Namespace

from load_newsgroups import load_20newsgroups
from logger_utils import setup_logger
from pubn import calculate_prior
from pubn.model import NlpBiasedLearner
from pubn.loss import LossType


def parse_args() -> Namespace:
    args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument("size_p", help="# elements in the POSITIVE (labeled) set", type=int)
    args.add_argument("size_n", help="# elements in the biased NEGATIVE (labeled) set", type=int)
    args.add_argument("loss", choices=[e.name for e in LossType])
    args.add_argument("--pos", help="List of class IDs for POSITIVE class", nargs='+', type=int)
    args.add_argument("--neg", help="List of class IDs for NEGATIVE class", nargs='+', type=int)

    args.add_argument("--ep", help="Number of training epochs", type=int,
                      default=NlpBiasedLearner.Config.NUM_EPOCH)
    args.add_argument("--bs", help="Batch size", type=int,
                      default=NlpBiasedLearner.Config.BATCH_SIZE)
    args.add_argument("--embed_dim", help="Word vector dimension", type=int,
                      default=NlpBiasedLearner.Config.EMBED_DIM)
    args.add_argument("--tau", help="Hyperparameter used to determine eta", type=float)

    args = args.parse_args()
    # Arguments error checking
    if args.size_p <= 0: raise ValueError("size_p must be positive valued")
    if args.ep <= 0: raise ValueError("Number of training epochs must be positive")
    NlpBiasedLearner.Config.NUM_EPOCH = args.ep
    if args.bs <= 0: raise ValueError("bs must be positive valued")
    NlpBiasedLearner.Config.BATCH_SIZE = args.bs
    if args.embed_dim <= 0: raise ValueError("Embedding vector dimension must be positive valued")
    NlpBiasedLearner.Config.EMBED_DIM = args.embed_dim

    # Configure any related fields
    args.loss = LossType[args.loss]

    args.pos, args.neg = set(args.pos), set(args.neg)
    assert not (args.pos & args.neg), "Positive and negative classes not disjoint"

    return args


def _main(args: Namespace):
    newsgroups = load_20newsgroups(args)  # ToDo fix 20 newsgroups to filter empty examples

    learner = NlpBiasedLearner(args, newsgroups.text.vocab.vectors,
                               prior=calculate_prior(newsgroups.test))
    learner.fit(newsgroups.train, newsgroups.label)


if __name__ == "__main__":
    setup_logger()
    _main(parse_args())
