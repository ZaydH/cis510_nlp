from pathlib import Path
import re
import socket

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
TORCH_DEVICE = torch.device("cuda:0" if IS_CUDA else "cpu")
if IS_CUDA:
    # noinspection PyUnresolvedReferences
    torch.set_default_dtype(torch.cuda.FloatTensor)

POS_LABEL = 1
U_LABEL = 0
NEG_LABEL = -1


def construct_iterator(ds: Dataset, bs: int, shuffle: bool = True) -> Iterator:
    r""" Construct \p Iterator which emulates a \p DataLoader """
    return Iterator(dataset=ds, batch_size=bs, shuffle=shuffle, device=TORCH_DEVICE)
