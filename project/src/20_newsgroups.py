import itertools
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import nltk.tokenize
import sklearn.datasets
from sklearn.utils import Bunch

import torchtext
import torchtext.datasets

# Valid Choices - Any subset of: ('headers', 'footers', 'quotes')
DATASET_REMOVE = ('headers', 'footers', 'quotes')
VALID_DATA_SUBSETS = ("train", "test", "all")

DATA_COL = "data"
LABEL_COL = "target"


def _get_20newsgroups(subset: str, data_dir: Path):
    r"""
    Gets the specified \p subset of the 20 Newsgroups dataset.  If necessary, the dataset is
    downloaded to directory \p data_dir.  It also tokenizes the import dataset

    :param data_dir: Path to the directory to sore
    :param subset: Valid choices, "train", "test", and "all"
    :return: Dataset
    """
    assert not data_dir.is_file(), "Must be a directory"
    assert subset in VALID_DATA_SUBSETS, "Invalid data subset"

    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = sklearn.datasets.fetch_20newsgroups(data_home=data_dir, shuffle=False,
                                                  remove=DATASET_REMOVE, subset=subset)

    def _tokenize(sentence: str) -> List[str]:
        return nltk.tokenize.word_tokenize(sentence.lower())

    dataset[DATA_COL] = [_tokenize(x) for x in dataset[DATA_COL]]
    return dataset


def _build_vocab(train: Bunch, test: Optional[Bunch]):
    r"""
    Gets all words in the vocabulary

    :param train: Training dataset
    :param test: Test dataset.  If no test set, use \p None
    :return: Set of words in the vocabulary.
    """
    words = set()
    for bunch in (train, test):
        if bunch is None: continue
        words |= {w for x in bunch[DATA_COL] for w in x}
    return words


def _filter_by_classes(bunch: Bunch, pos_classes: Set[int],
                       neg_classes: Set[int]) -> List[Tuple[int, List[str]]]:
    r""" Removes any items not in the specified class label list """
    all_classes = pos_classes | neg_classes
    # Filter according to positive and negative class set
    keep_idx = [val in all_classes for val in bunch[LABEL_COL]]
    assert any(keep_idx), "No elements to keep list"

    def _filt_col(col_name: str):
        return list(itertools.compress(bunch[col_name], keep_idx))

    data_filt, lbl_filt = _filt_col(DATA_COL), _filt_col(LABEL_COL)
    assert len(data_filt) == len(lbl_filt), "Data/label array length mismatch"

    return list(zip(lbl_filt, data_filt))


def load_20newsgroups(data_dir: Union[Path, str], pos_classes: Set[int], neg_classes: Set[int]):
    r"""

    :param data_dir: Directory from which to get the training data
    :param pos_classes: Set of labels for the POSITIVE classes
    :param neg_classes: Set of labels for the NEGATIVE classes
    :return: Train and test datasets
    """
    assert pos_classes and neg_classes, "Class list empty"

    data_dir = Path(data_dir)

    train, test = _get_20newsgroups("train", data_dir), _get_20newsgroups("test", data_dir)
    vocab = _build_vocab(train=train, test=test)

    all_labels = pos_classes | neg_classes
    ds = []
    for bunch in (train, test):
        filt_ds = _filter_by_classes(bunch, pos_classes, neg_classes)
        ds.append(torchtext.datasets.TextClassificationDataset(vocab, filt_ds, all_labels))
    return ds[0], ds[1]


def _get_field():
    import torchtext.data
    f = torchtext.data.field


if __name__ == "__main__":
    load_20newsgroups(".", set(range(0, 10)), set(range(10, 20)))
