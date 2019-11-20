from argparse import Namespace
from enum import Enum
from pathlib import Path
import re
import socket
from typing import Set

import torch
from torchtext.data import Dataset, Iterator


def _check_is_talapas() -> bool:
    r""" Returns \p True if running on talapas """
    host = socket.gethostname().lower()
    if "talapas" in host:
        return True
    if re.match(r"^n\d{3}$", host):
        return True

    num_string = r"(\d{3}|\d{3}-\d{3})"
    if re.match(f"n\\[{num_string}(,{num_string})*\\]", host):
        return True
    return False


IS_TALAPAS = _check_is_talapas()
BASE_DIR = Path(".").absolute() if not IS_TALAPAS else Path("/home/zhammoud/projects/nlp")

IS_CUDA = torch.cuda.is_available()
if IS_CUDA:
    # noinspection PyUnresolvedReferences
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
TORCH_DEVICE = torch.device("cuda:0" if IS_CUDA else "cpu")

POS_LABEL = 1
U_LABEL = 0
NEG_LABEL = -1


def construct_iterator(ds: Dataset, bs: int, shuffle: bool = True) -> Iterator:
    r""" Construct \p Iterator which emulates a \p DataLoader """
    return Iterator(dataset=ds, batch_size=bs, shuffle=shuffle, device=TORCH_DEVICE)


def construct_filename(prefix: str, args: Namespace, out_dir: Path, file_ext: str) -> Path:
    r""" Standardize naming scheme for the filename """

    def _classes_to_str(cls_set: Set[Enum]) -> str:
        return ",".join([x.name.lower() for x in sorted(cls_set)])

    fields = [prefix] if prefix else []
    fields += [f"n-p={args.size_p}", f"n-n={args.size_n}", f"n-u={args.size_u}",
               f"pos={_classes_to_str(args.pos)}", f"neg={_classes_to_str(args.neg)}",
               f"seq={args.seq_len}"]

    if args.bias:
        # Ensure bias has same order as
        bias_sorted = [x for _, x in sorted(zip(args.neg, args.bias))]
        fields.append(f"bias={','.join([f'{x:02}' for x in bias_sorted])}")

    if file_ext[0] != ".": file_ext = "." + file_ext
    fields[-1] += file_ext

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "_".join(fields)
