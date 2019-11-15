import argparse
from argparse import Namespace

from pu_loss import LossType


def parse_args() -> Namespace:
    args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument("loss", choices=[e.name for e in LossType])
    args.add_argument("--tau", help="Hyperparameter used to determine eta", type=float)

    args = args.parse_args()
    args.loss = LossType[args.loss]
    return args


def _main(args: Namespace):


if __name__ == "__main__":
    _main(parse_args())
