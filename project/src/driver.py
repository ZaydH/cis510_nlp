from argparse import Namespace

import load_20newsgroups
from generate_results import calculate_results
from input_args import parse_args
from logger_utils import setup_logger

from pubn.model import NlpBiasedLearner


def _main(args: Namespace):
    ngd = load_20newsgroups.load(args)

    embedding = None if args.preprocess else ngd.text.vocab.vectors
    classifier = NlpBiasedLearner(args, embedding, prior=ngd.prior)
    # noinspection PyUnresolvedReferences
    classifier.fit(train=ngd.train, valid=ngd.valid, unlabel=ngd.unlabel,
                   label=None if args.preprocess else ngd.label)

    calculate_results(args, classifier, ngd.label, unlabel_ds=ngd.unlabel, test_ds=ngd.test)


if __name__ == "__main__":
    setup_logger()
    _main(parse_args())
