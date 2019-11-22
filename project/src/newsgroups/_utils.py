from argparse import Namespace
from pathlib import Path
from typing import Optional, Union

from ._load_iterator import load_newsgroups_iterator


def load(args: Namespace, preprocessed_dir: Optional[Union[Path, str]] = None):
    if preprocessed_dir is None:
        return load_newsgroups_iterator(args)

    preprocessed_dir = Path(preprocessed_dir)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    return load_newsgroups_preprocessed(args)
