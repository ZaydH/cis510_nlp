from argparse import Namespace
import copy
from dataclasses import dataclass
import itertools
import logging
from enum import Enum
from pathlib import Path
import pickle as pk
from typing import Optional, Set, Tuple

import nltk.tokenize
import numpy as np
import sklearn.datasets
# noinspection PyProtectedMember
from sklearn.utils import Bunch

import torchtext
from torchtext.data import Example, Field, LabelField
import torchtext.datasets
from torchtext.data.dataset import Dataset
import torchtext.vocab

# Valid Choices - Any subset of: ('headers', 'footers', 'quotes')
from logger_utils import setup_logger
from pubn import BASE_DIR, NEG_LABEL, POS_LABEL, U_LABEL
from pubn.loss import LossType

# DATASET_REMOVE = ('headers', 'footers', 'quotes')  # ToDo settle on dataset elements to remove
DATASET_REMOVE = ('headers', 'footers')
VALID_DATA_SUBSETS = ("train", "test", "all")

DATA_COL = "data"
LABEL_COL = "target"
LABEL_NAMES_COL = "target_names"

# MAX_SEQ_LEN = 500  # ToDo fix max sequence length
MAX_SEQ_LEN = 50


@dataclass(init=True)
class NewsgroupsData:
    r""" Encapsulates the 20 newsgroups dataset """
    text: Field
    label: LabelField
    train: Dataset = None
    test: Dataset = None
    unlabel: Dataset = None

    class Categories(Enum):
        ALT = {0}
        COMP = {1, 2, 3, 4, 5}
        MISC = {6}
        REC = {7, 8, 9, 10}
        SCI = {11, 12, 13, 14}
        SOC = {15}
        TALK = {16, 17, 18, 19}

        def __lt__(self, other: 'NewsgroupsData.Categories') -> bool:
            return min(self.value) < min(other.value)

    @staticmethod
    def _pickle_filename(args: Namespace) -> Path:
        r""" File name for pickle file """
        serialize_dir = BASE_DIR / "tensors"
        serialize_dir.mkdir(parents=True, exist_ok=True)

        def _classes_to_str(cls_set: 'Set[NewsgroupsData.Categories]') -> str:
            return ",".join([x.name.lower() for x in sorted(cls_set)])

        fields = ["data", f"n-p={args.size_p}", f"n-n={args.size_n}",
                  f"pos={_classes_to_str(args.pos)}", f"neg={_classes_to_str(args.neg)}"]

        fields[-1] += ".pk"
        return serialize_dir / "_".join(fields)

    @classmethod
    def serial_exists(cls, args: Namespace) -> bool:
        r""" Return \p True if a serialized dataset exists for the configuration in \p args """
        serial_path = cls._pickle_filename(args)
        return serial_path.exists()

    def dump(self, args: Namespace):
        r""" Serialize the newsgroup data to disk """
        path = self._pickle_filename(args)

        msg = f"Writing serialized file {str(path)}"
        flds = (self.text, self.label, self.train.examples, self.test.examples,
                self.unlabel.examples)
        logging.debug(f"Starting: {msg}")
        with open(str(path), "wb+") as f_out:
            pk.dump(flds, f_out)
        logging.debug(f"COMPLETED: {msg}")

    def build_fields(self):
        r""" Construct the dataset fields """
        return [("text", self.text), ("label", self.label)]

    @classmethod
    def load(cls, args: Namespace):
        r""" Load the serialized newsgroups dataset """
        path = cls._pickle_filename(args)

        with open(str(path), "rb") as f_in:
            flds = pk.load(f_in)
        newsgroup = cls(text=flds[0], label=flds[1])
        for attr_name, idx in (("train", 2), ("test", 3), ("unlabel", 4)):
            newsgroup.__setattr__(attr_name, Dataset(flds[idx], newsgroup.build_fields()))
        return newsgroup


def _download_20newsgroups(subset: str, data_dir: Path, pos_cls: Set[int], neg_cls: Set[int]):
    r"""
    Gets the specified \p subset of the 20 Newsgroups dataset.  If necessary, the dataset is
    downloaded to directory \p data_dir.  It also tokenizes the import dataset

    :param data_dir: Path to the directory to sore
    :param subset: Valid choices, "train", "test", and "all"
    :return: Dataset
    """
    msg = f"Download {subset} 20 newsgroups dataset"
    logging.debug(f"Starting: {msg}")

    assert not data_dir.is_file(), "Must be a directory"
    assert subset in VALID_DATA_SUBSETS, "Invalid data subset"

    data_dir.mkdir(parents=True, exist_ok=True)
    bunch = sklearn.datasets.fetch_20newsgroups(data_home=data_dir, shuffle=False,
                                                  remove=DATASET_REMOVE, subset=subset)
    _filter_bunch_by_classes(bunch, cls_to_keep=pos_cls | neg_cls)

    logging.debug(f"COMPLETED: {msg}")
    return bunch


def _filter_bunch_by_classes(bunch: Bunch, cls_to_keep: Set[int]):
    r""" Removes any dataset items not in the specified class label lists """
    keep_idx = [val in cls_to_keep for val in bunch[LABEL_COL]]
    assert any(keep_idx), "No elements to keep list"

    for key, val in bunch.items():
        if not isinstance(val, (list, np.ndarray)): continue
        if len(val) != len(keep_idx): continue
        bunch[key] = list(itertools.compress(val, keep_idx))


# def filter_dataset_by_cls(ds: Dataset, label_to_remove: Union[Set[int], int]) -> Subset:
#     r""" Select a subset of the \p Dataset at the exclusion """
#     if isinstance(label_to_remove, int): label_to_remove = {label_to_remove}
#
#     indices = []
#     for idx, x in enumerate(ds):
#         if x.label in label_to_remove:
#             indices.append(idx)
#
#     # indices = torch.tensor(indices, dtype=torch.int64)
#     return Subset(ds, indices)


def _filter_bunch_by_idx(bunch: Bunch, keep_idx):
    r"""
    Filters \p Bunch object and removes any unneeded elements

    :param bunch: Dataset \p Bunch object to filter
    :param keep_idx: List of Boolean values where the value is \p True if the element should be
                     kept.
    :return: Filtered \p Bunch object
    """
    bunch = copy.deepcopy(bunch)
    for key, val in bunch.items():
        if not isinstance(val, (list, np.ndarray)): continue
        if len(keep_idx) != len(val): continue

        bunch[key] = list(itertools.compress(val, keep_idx))
        if isinstance(val, np.ndarray): bunch[key] = np.asarray(bunch[key])
    return bunch


def _configure_binary_labels(bunch: Bunch, pos_cls: Set[int]):
    r""" Takes the 20 Newsgroup labels and binarizes them """
    def _is_pos(lbl: int) -> int:
        return POS_LABEL if lbl in pos_cls else NEG_LABEL

    bunch[LABEL_COL] = np.asarray(list(map(_is_pos, bunch[LABEL_COL])), dtype=np.int64)


def _select_positive_bunch(size_p: int, bunch: Bunch, pos_cls: Set[int],
                           remove_p_from_u: bool) -> Tuple[Bunch, Bunch]:
    r"""

    :param size_p: Number of (positive elements) to select from Bunch
    :param bunch: Training (unlabeled) \p Bunch
    :param pos_cls: List of positive classes in t
    :param remove_p_from_u: If \p True, elements in the positive (labeled) dataset are removed
                            from the unlabeled set.
    :return:
    """
    pos_idx = np.asarray([idx for idx, lbl in enumerate(bunch[LABEL_COL]) if lbl in pos_cls],
                         dtype=np.int32)

    assert len(pos_idx) >= size_p, "P set larger than the available data"
    np.random.shuffle(pos_idx)
    keep_idx = np.zeros_like(bunch[LABEL_COL], dtype=np.bool)
    for idx in pos_idx[:size_p]: keep_idx[idx] = True

    p_bunch = _filter_bunch_by_idx(bunch, keep_idx)
    p_bunch[LABEL_NAMES_COL] = [bunch[LABEL_NAMES_COL][idx] for idx in sorted(list(pos_cls))]
    if remove_p_from_u:
        bunch = _filter_bunch_by_idx(bunch, ~keep_idx)
    return p_bunch, bunch


def _bunch_to_ds(bunch: Bunch, text: Field, label: LabelField) -> Dataset:
    r""" Converts the \p bunch to a classification dataset """
    fields = [('text', text), ('label', label)]
    examples = [Example.fromlist(x, fields) for x in zip(bunch[DATA_COL], bunch[LABEL_COL])]
    return Dataset(examples, fields)


def _print_stats(text: Field, label: LabelField):
    r""" Log information about the dataset as a sanity check """
    logging.info(f"Maximum sequence length: {MAX_SEQ_LEN}")
    logging.info(f"Length of Text Vocabulary: {str(len(text.vocab))}")
    logging.info(f"Vector size of Text Vocabulary: {text.vocab.vectors.shape[1]}")
    logging.info("Label Length: " + str(len(label.vocab)))


def _build_train_set(p_bunch: Bunch, u_bunch: Bunch, n_bunch: Optional[Bunch],
                     text: Field, label: LabelField) -> Dataset:
    r"""
    Convert the positive, negative, and unlabeled \p Bunch objects into a Dataset
    """
    data, labels, names = [], [], []
    for bunch, lbl in ((p_bunch, POS_LABEL), (u_bunch, U_LABEL), (n_bunch, NEG_LABEL)):
        if bunch is None: continue
        data.extend(bunch[DATA_COL])
        labels.append(np.full_like(bunch[LABEL_COL], lbl))

    t_bunch = copy.deepcopy(u_bunch)
    t_bunch[DATA_COL] = data
    t_bunch[LABEL_COL] = np.concatenate(labels, axis=0)
    return _bunch_to_ds(t_bunch, text, label)


def _create_serialized_20newsgroups(args):
    r"""
    Creates a serialized 20 newsgroups dataset

    :param args: Test setup information
    """
    data_dir = BASE_DIR / ".data"
    cache_dir = data_dir / ".vector_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    p_cls = {cls_id for cls_grp in args.pos for cls_id in cls_grp.value}
    n_cls = {cls_id for cls_grp in args.neg for cls_id in cls_grp.value}
    complete_train = _download_20newsgroups("train", data_dir, p_cls, n_cls)

    tokenizer = nltk.tokenize.word_tokenize
    # noinspection PyPep8Naming
    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True,
                 fix_length=MAX_SEQ_LEN)
    # noinspection PyPep8Naming
    LABEL = LabelField(sequential=False)
    complete_ds = _bunch_to_ds(complete_train, TEXT, LABEL)
    TEXT.build_vocab(complete_ds, min_freq=2,
                     vectors=torchtext.vocab.GloVe(name="6B", dim=args.embed_dim, cache=cache_dir))

    p_bunch, u_bunch = _select_positive_bunch(args.size_p, complete_train, p_cls,
                                              remove_p_from_u=False)
    n_bunch = None  # ToDo Add code for getting the N bunch

    test_bunch = _download_20newsgroups("test", data_dir, p_cls, n_cls)

    # Binarize the labels
    for bunch in (p_bunch, u_bunch, test_bunch):
        _configure_binary_labels(bunch, pos_cls=p_cls)

    ng_data = NewsgroupsData(text=TEXT, label=LABEL)
    ng_data.train = _build_train_set(p_bunch,
                                     u_bunch if args.loss != LossType.PN else None,
                                     n_bunch if args.loss != LossType.NNPU else None,
                                     TEXT, LABEL)
    ng_data.unlabel = _bunch_to_ds(u_bunch, TEXT, LABEL)
    ng_data.test = _bunch_to_ds(test_bunch, TEXT, LABEL)

    LABEL.build_vocab(ng_data.train, ng_data.test)
    ng_data.dump(args)


def load_20newsgroups(args: Namespace):
    r"""
    Automatically downloads the 20 newsgroups dataset.
    :param args: Parsed command line arguments
    """
    assert args.pos and args.neg, "Class list empty"
    assert not (args.pos & args.neg), "Positive and negative classes not disjoint"

    # Load the serialized file if it exists
    if not NewsgroupsData.serial_exists(args):
        _create_serialized_20newsgroups(args)
    # _create_serialized_20newsgroups(serialize_path, args)

    serial = NewsgroupsData.load(args)
    _print_stats(serial.text, serial.label)
    return serial


def _main():
    args = Namespace()
    args.size_p = 1000
    args.size_n = 0
    args.embed_dim = 300
    args.pos, args.neg = set(range(0, 10)),  set(range(10, 20))

    # noinspection PyUnusedLocal
    newsgroups = load_20newsgroups(args)

    print("Completed")


if __name__ == "__main__":
    setup_logger()
    _main()
