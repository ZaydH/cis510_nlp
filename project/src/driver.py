import argparse
from argparse import Namespace

from torchnet.dataset.dataset import Dataset

from load_newsgroups import load_20newsgroups, filter_dataset_by_cls, construct_iterator
from pubn import NEG_LABEL, U_LABEL
from pubn.model import NlpBiasedLearner
from pubn.loss import LossType


def parse_args() -> Namespace:
    args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument("size_p", help="# elements in the POSITIVE (labeled) set", type=int)
    args.add_argument("size_n", help="# elements in the biased NEGATIVE (labeled) set", type=int)
    args.add_argument("loss", choices=[e.name for e in LossType])
    args.add_argument("--pos", nargs='+', type=int)
    args.add_argument("--neg", nargs='+', type=int)
    args.add_argument("--bs", help="Batch size", type=int, default=32)
    args.add_argument("--embed_dim", help="Word vector dimension", type=int, default=300)
    args.add_argument("--tau", help="Hyperparameter used to determine eta", type=float)

    args = args.parse_args()
    # Arguments error checking
    if args.size_p <= 0: raise ValueError("size_p must be positive valued")
    if args.bs <= 0: raise ValueError("bs must be positive valued")
    if args.embed_dim <= 0: raise ValueError("Embedding vector dimension must be positive valued")

    # Configure any related fields
    args.loss = LossType[args.loss]

    args.pos, args.neg = set(args.pos), set(args.neg)
    assert not (args.pos & args.neg), "Positive and negative classes not disjoint"

    NlpBiasedLearner.Config.EMBED_DIM = args.embed_dim

    return args


def _train_learner(args: Namespace, learner: NlpBiasedLearner, train_ds: Dataset):
    if args.loss == LossType.NNPU or args.loss == LossType.PN:
        if args.loss == LossType.NNPU:
            label_to_remove = NEG_LABEL
        elif args.loss == LossType.PN:
            label_to_remove = U_LABEL
        else:
            raise ValueError("Unable to parse ")
        train_ds = filter_dataset_by_cls(train_ds, label_to_remove)

    itr = construct_iterator(train_ds, bs=args.bs, shuffle=True)
    learner.fit(itr)


def _main(args: Namespace):
    TEXT, LABEL, train_ds, test_ds = load_20newsgroups(args)

    learner = NlpBiasedLearner(args, TEXT.vocab.vectors)
    _train_learner(args, learner, train_ds)


if __name__ == "__main__":
    _main(parse_args())
