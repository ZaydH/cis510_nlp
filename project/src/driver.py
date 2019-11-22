from argparse import Namespace

from generate_results import calculate_results
from input_args import parse_args
import newsgroups
from logger_utils import setup_logger

from pubn import DATA_DIR
from pubn.model import NlpBiasedLearner


def _main(args: Namespace):
    preprocessed_dir = (DATA_DIR / "preprocessed") if args.preprocess else None
    ngd = newsgroups.load(args, preprocessed_dir)

    classifier = NlpBiasedLearner(args, ngd.text.vocab.vectors, prior=ngd.prior)
    # noinspection PyUnresolvedReferences
    classifier.fit(train=ngd.train, valid=ngd.valid, unlabel=ngd.unlabel, label=ngd.label)

    calculate_results(args, classifier, ngd.label, unlabel_ds=ngd.unlabel, test_ds=ngd.test)


if __name__ == "__main__":
    setup_logger()
    _main(parse_args())
