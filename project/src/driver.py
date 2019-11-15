import argparse
from argparse import Namespace

from pu_loss import LossType


def parse_args() -> Namespace:
    args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument("size_p", "Number of elements in the positive set", type=int)
    args.add_argument("loss", choices=[e.name for e in LossType])
    args.add_argument("--tau", help="Hyperparameter used to determine eta", type=float)

    args = args.parse_args()
    if args.size_p <= 0:
        raise ValueError("size_p must be positive valued")
    args.loss = LossType[args.loss]
    return args


def _main(args: Namespace):
    pass

if __name__ == "__main__":
    _main(parse_args())
