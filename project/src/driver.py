from argparse import Namespace

from generate_results import calculate_results
from input_args import parse_args
from load_newsgroups import load_20newsgroups
from logger_utils import setup_logger
from pubn import calculate_prior
from pubn.model import NlpBiasedLearner


def _main(args: Namespace):
    newsgroups = load_20newsgroups(args)  # ToDo fix 20 newsgroups to filter empty examples

    classifier = NlpBiasedLearner(args, newsgroups.text.vocab.vectors,
                                  prior=calculate_prior(newsgroups.test))
    # noinspection PyUnresolvedReferences
    classifier.fit(newsgroups.train, newsgroups.train, newsgroups.label)  # ToDo Fix valid ds

    calculate_results(args, classifier, newsgroups.label, unlabel_ds=newsgroups.unlabel,
                      test_ds=newsgroups.test)


if __name__ == "__main__":
    setup_logger()
    _main(parse_args())
