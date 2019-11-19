from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

from load_newsgroups import NewsgroupsData
from pubn.loss import LossType
from pubn.model import NlpBiasedLearner


def parse_args() -> Namespace:
    args = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    args.add_argument("size_p", help="# elements in the POSITIVE (labeled) set", type=int)
    args.add_argument("size_n", help="# elements in the biased NEGATIVE (labeled) set", type=int)
    args.add_argument("size_u", help="# elements in the UNLABELED set", type=int)
    args.add_argument("loss", help="Loss type to use", choices=[x.name.lower() for x in LossType])
    args.add_argument("--pos", help="List of class IDs for POSITIVE class", nargs='+', type=str,
                      choices=[e.name.lower() for e in NewsgroupsData.Categories])
    args.add_argument("--neg", help="List of class IDs for NEGATIVE class", nargs='+', type=str,
                      choices=[e.name.lower() for e in NewsgroupsData.Categories])
    msg = ("Bias vector for the negative class (optional). \n. If specified, must be the same "
           "length as the negative class list.  Values are automatically normalized to sum to 1")
    args.add_argument("--bias", help=msg, nargs='*', type=float)

    args.add_argument("--rho", help="Pr[y=-1, s=+1]", type=float, default=None)
    args.add_argument("--ep", help="Number of training epochs", type=int,
                      default=NlpBiasedLearner.Config.NUM_EPOCH)
    args.add_argument("--bs", help="Batch size", type=int,
                      default=NlpBiasedLearner.Config.BATCH_SIZE)
    args.add_argument("--embed_dim", help="Word vector dimension", type=int,
                      default=NlpBiasedLearner.Config.EMBED_DIM)
    args.add_argument("--seq_len", help="Maximum sequence length",  type=int, default=500)
    args.add_argument("--tau", help="Hyperparameter used to determine eta", type=float)

    args = args.parse_args()

    _error_check_args(args)
    _refactor_args(args)
    _transfer_args_to_config(args)
    return args


def _error_check_args(args: Namespace):
    # Arguments error checking
    pos_flds = ("size_p", "size_n", "size_n", "bs", "ep", "embed_dim")
    for name in pos_flds:
        if args.__getattribute__(name) <= 0: raise ValueError(f"{name} must be positive valued")

    if set(args.pos) & set(args.neg):
        raise ValueError("Positive and negative classes not disjoint")

    if args.bias:
        if len(args.bias) != len(args.neg):
            raise ValueError("Bias and negative vector length mismatch")
        if any(x < 0 for x in args.bias):
            raise ValueError("Bias values must be non-negative")
        if sum(args.bias):
            raise ValueError("No positive bias probabilities specified")

    if args.rho is not None:
        if args.loss != LossType.PUBN.name.lower():
            raise ValueError(f"rho specified but not valid for loss \"{args.loss}\"")
        if args.rho <= 0 or args.rho >= 1:
            raise ValueError(f"rho must be in the range (0,1)")
    else:
        if args.loss == LossType.PUbN.name.lower():
            raise ValueError("rho not specified but PUbN used")


def _refactor_args(args: Namespace) -> None:
    r""" Reformat any arguments from the raw inputs into more usable forms """
    args.loss = LossType[args.loss.upper()]

    # Convert 20 newsgroups group names to actual objects
    for ds_name in ("pos", "neg"):
        val = [NewsgroupsData.Categories[x.upper()] for x in args.__getattribute__(ds_name)]
        args.__setattr__(ds_name, val)

    if args.bias:
        # noinspection PyTypeChecker
        bias_vec = [x / sum(args.bias) for x in args.bias]  # Normalize the total bias to 1
        args.bias = [(cls, bias) for cls, bias in zip(args.neg, bias_vec) if bias > 0]

    # Must convert to sets after bias creation to ensure 1-to-1 mapping
    args.pos, args.neg = set(args.pos), set(args.neg)


def _transfer_args_to_config(args: Namespace):
    r""" Transfer the values in args to any configuration"""
    NlpBiasedLearner.Config.NUM_EPOCH = args.ep
    NlpBiasedLearner.Config.BATCH_SIZE = args.bs
    NlpBiasedLearner.Config.EMBED_DIM = args.embed_dim
