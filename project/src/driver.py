import argparse

from model import LossType


def parse_args():
    args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument("loss", choices=[e.name for e in LossType])
    args.add_argument("--tau", help="Hyperparameter used to determine eta", type=float)

    args.
