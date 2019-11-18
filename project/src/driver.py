import argparse
from argparse import Namespace

from generate_results import calculate_results
from load_newsgroups import NewsgroupsData, load_20newsgroups
from logger_utils import setup_logger
from pubn import calculate_prior
from pubn.model import NlpBiasedLearner
from pubn.loss import LossType


def parse_args() -> Namespace:
    args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument("size_p", help="# elements in the POSITIVE (labeled) set", type=int)
    args.add_argument("size_n", help="# elements in the biased NEGATIVE (labeled) set", type=int)
    args.add_argument("size_u", help="# elements in the UNLABELED set", type=int)
    args.add_argument("loss", help="Loss type to use", choices=[x.name.lower() for x in LossType])
    args.add_argument("--pos", help="List of class IDs for POSITIVE class", nargs='+', type=str,
                      choices=[e.name.lower() for e in NewsgroupsData.Categories])
    args.add_argument("--neg", help="List of class IDs for NEGATIVE class", nargs='+', type=str,
                      choices=[e.name.lower() for e in NewsgroupsData.Categories])

    args.add_argument("--ep", help="Number of training epochs", type=int,
                      default=NlpBiasedLearner.Config.NUM_EPOCH)
    args.add_argument("--bs", help="Batch size", type=int,
                      default=NlpBiasedLearner.Config.BATCH_SIZE)
    args.add_argument("--embed_dim", help="Word vector dimension", type=int,
                      default=NlpBiasedLearner.Config.EMBED_DIM)
    args.add_argument("--tau", help="Hyperparameter used to determine eta", type=float)

    args = args.parse_args()
    # Arguments error checking
    pos_flds = ("size_p", "size_n", "size_n", "bs", "ep", "embed_dim")
    for name in pos_flds:
        if args.__getattribute__(name) <= 0: raise ValueError(f"{name} must be positive valued")

    NlpBiasedLearner.Config.NUM_EPOCH = args.ep
    NlpBiasedLearner.Config.BATCH_SIZE = args.bs
    NlpBiasedLearner.Config.EMBED_DIM = args.embed_dim

    # Configure any related fields
    args.loss = LossType[args.loss.upper()]

    args.pos = {NewsgroupsData.Categories[x.upper()] for x in args.pos}
    args.neg = {NewsgroupsData.Categories[x.upper()] for x in args.neg}
    assert not (args.pos & args.neg), "Positive and negative classes not disjoint"

    return args


def _main(args: Namespace):
    newsgroups = load_20newsgroups(args)  # ToDo fix 20 newsgroups to filter empty examples

    classifier = NlpBiasedLearner(args, newsgroups.text.vocab.vectors,
                                  prior=calculate_prior(newsgroups.test))
    # noinspection PyUnresolvedReferences
    classifier.fit(newsgroups.train, newsgroups.label)

    calculate_results(args, classifier, newsgroups.label, unlabel_ds=newsgroups.unlabel,
                      test_ds=newsgroups.test)


if __name__ == "__main__":
    setup_logger()
    _main(parse_args())
