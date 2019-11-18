from argparse import Namespace
import copy
from dataclasses import dataclass
from enum import Enum
import itertools
import logging
from pathlib import Path
import pickle as pk
from typing import List, Optional, Set, Tuple

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
from pubn import BASE_DIR, NEG_LABEL, POS_LABEL, U_LABEL, construct_filename

# DATASET_REMOVE = ('headers', 'footers', 'quotes')  # ToDo settle on dataset elements to remove
DATASET_REMOVE = ('headers', 'footers')
VALID_DATA_SUBSETS = ("train", "test", "all")

DATA_COL = "data"
LABEL_COL = "target"
LABEL_NAMES_COL = "target_names"


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
        return construct_filename("data", args, serialize_dir, "pk")

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
    all_cls = pos_cls | neg_cls
    keep_idx = [val in all_cls for val in bunch[LABEL_COL]]
    assert any(keep_idx), "No elements to keep list"

    for key, val in bunch.items():
        if not isinstance(val, (list, np.ndarray)): continue
        if len(val) != len(keep_idx): continue
        bunch[key] = list(itertools.compress(val, keep_idx))

    logging.debug(f"COMPLETED: {msg}")
    return bunch


def _select_indexes_uar(orig_size: int, new_size: int) -> np.ndarray:
    r"""
    Selects a set of indices uniformly at random (uar) without replacement.
    :param orig_size: Original size of the array
    :param new_size: New size of the array
    :return: Boolean list of size \p original_size where \p True represents index selected
    """
    shuffled = np.arange(orig_size)
    np.random.shuffle(shuffled)
    keep_idx = np.zeros_like(shuffled, dtype=np.bool)
    for i in np.arange(new_size):
        keep_idx[shuffled[i]] = True
    return keep_idx


def _reduce_to_fixed_size(bunch: Bunch, new_size: int):
    r""" Reduce the bunch to a fixed size """
    orig_size = len(bunch[LABEL_COL])
    assert orig_size >= new_size

    keep_idx = _select_indexes_uar(orig_size, new_size)
    return _filter_bunch_by_idx(bunch, keep_idx)


def _filter_bunch_by_idx(bunch: Bunch, keep_idx: np.ndarray):
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


def _configure_binary_labels(bunch: Bunch, pos_cls: Set[int], neg_cls: Set[int]):
    r""" Takes the 20 Newsgroup labels and binarizes them """
    def _is_pos(lbl: int) -> int:
        if lbl in pos_cls: return POS_LABEL
        if lbl in neg_cls: return NEG_LABEL
        raise ValueError(f"Unknown label {lbl}")

    bunch[LABEL_COL] = np.asarray(list(map(_is_pos, bunch[LABEL_COL])), dtype=np.int64)


def _convert_selected_idx_to_keep_list(sel_idx: np.ndarray, keep_list_size: int) -> np.ndarray:
    r"""
    Converts the list of integers into a binary vector with the specified indices \p True.

    :param sel_idx: Indices of the return vector to be \p True.
    :param keep_list_size: Size of the Boolean vector to return
    :return: Boolean vector with integers in \p sel_idx \p True and otherwise \p False
    """
    assert keep_list_size > max(sel_idx), "Invalid size for the keep list"
    keep_idx = np.zeros((keep_list_size,), dtype=np.bool)
    for idx in sel_idx:
        keep_idx[idx] = True
    return keep_idx


def _get_idx_of_classes(bunch: Bunch, cls_ids: Set[int]) -> np.ndarray:
    r""" Returns index of all examples in \p Bunch whose label is in \p cls_ids """
    return np.asarray([idx for idx, lbl in enumerate(bunch[LABEL_COL]) if lbl in cls_ids],
                       dtype=np.int32)


def _select_items_from_bunch(bunch: Bunch, selected_cls: Set[int], selected_idx: np.ndarray,
                             remove_sel_from_bunch: bool) -> Tuple[Bunch, Bunch]:
    r"""
    Selects a set of items (given by indices in \p selected_idx) and returns it as a new \p Bunch.
    Optionally filters the input \p bunch to remove the selected items.

    :param bunch: Bunch to select from
    :param selected_cls: Class ID numbers for the selected items
    :param selected_idx: Index of the elements in \p bunch to select
    :param remove_sel_from_bunch: If \p True, removed selected indexes from \p bunch and return it
    :return: Selected \p Bunch and other \p Bunch optionally filtered
    """
    assert len(bunch[LABEL_COL]) > max(selected_idx), "Invalid size for the keep list"

    keep_idx = _convert_selected_idx_to_keep_list(selected_idx, len(bunch[LABEL_COL]))

    sel_bunch = _filter_bunch_by_idx(bunch, keep_idx)
    sel_bunch[LABEL_NAMES_COL] = [bunch[LABEL_NAMES_COL][idx] for idx in sorted(list(selected_cls))]

    # Sanity check no unexpected classes in selected bunch
    assert all(x in selected_cls for x in sel_bunch[LABEL_COL]), "Invalid selected class in bunch"

    if remove_sel_from_bunch:
        return sel_bunch, _filter_bunch_by_idx(bunch, ~keep_idx)
    return sel_bunch, bunch


def _select_bunch_uar(size: int, bunch: Bunch, cls_ids: Set[int],
                      remove_from_bunch: bool) -> Tuple[Bunch, Bunch]:
    r"""
    Selects elements with a class label in cls_ids uniformly at random (uar) without replacement
    from \p bunch.  Optionally removes those elements from \p bunch as well.

    :param size: Number of (positive elements) to select from Bunch
    :param bunch: Source \p Bunch
    :param cls_ids: List of classes in the selected \p Bunch
    :param remove_from_bunch: If \p True, elements in the selected bunch are removed from \p bunch.
    :return: Tuple of the selected bunch and the other bunch (optionally filtered)
    """
    cls_idx = _get_idx_of_classes(bunch, cls_ids)
    sel_keep_idx = _select_indexes_uar(len(cls_idx), size)
    sel_idx = np.array(list(itertools.compress(cls_idx, sel_keep_idx)))

    return _select_items_from_bunch(bunch, cls_ids, sel_idx, remove_from_bunch)


def _select_negative_bunch(size_n: int, bunch: Bunch, neg_cls: Set[int],
                           bias: Optional[List[Tuple[NewsgroupsData.Categories, float]]],
                           remove_from_bunch: bool) -> Tuple[Bunch, Bunch]:
    r"""
    Randomly selects a negative bunch of size \p size_n.  If \p bias is \p None, the negative bunch 
    is selected u.a.r. from all class IDs in \p neg_cls.  Otherwise, probability each group is
    selected is specified by the \p bias vector.  Optionally removes the selected elements
    from \p bunch.
    
    :param size_n:  Size of new negative set.
    :param bunch: Bunch from which to select the negative elements.
    :param neg_cls: ID numbers for the negative set
    :param bias: Optional vector for bias
    :param remove_from_bunch: If \p True, elements in the selected bunch are removed from \p bunch.
    :return: Tuple of the selected bunch and the other bunch (optionally filtered)
    """
    # If no bias, select the elements u.a.r.
    if bias is None:
        return _select_bunch_uar(size_n, bunch, neg_cls, remove_from_bunch)

    # Multinomial distribution from Pr[x|y=-1,s =+1]
    grp_sizes = np.random.multinomial(size_n, [prob for _, prob in bias])
    # Determine selected index
    sel_idx = []
    for (cls_lst, _), num_ele in zip(bias, grp_sizes):
        cls_idx = _get_idx_of_classes(bunch, cls_lst.value)
        assert len(cls_idx) >= num_ele, "Insufficient elements in list"
        keep_idx = _select_indexes_uar(len(cls_idx), num_ele)
        sel_idx.append(np.array(list(itertools.compress(cls_idx, keep_idx))))

    sel_idx = np.concatenate(sel_idx, axis=0)
    return _select_items_from_bunch(bunch, neg_cls, sel_idx, remove_from_bunch)


def _bunch_to_ds(bunch: Bunch, text: Field, label: LabelField) -> Dataset:
    r""" Converts the \p bunch to a classification dataset """
    fields = [('text', text), ('label', label)]
    examples = [Example.fromlist(x, fields) for x in zip(bunch[DATA_COL], bunch[LABEL_COL])]
    return Dataset(examples, fields)


def _print_stats(text: Field, label: LabelField):
    r""" Log information about the dataset as a sanity check """
    logging.info(f"Maximum sequence length: {text.fix_length}")
    logging.info(f"Length of Text Vocabulary: {str(len(text.vocab))}")
    logging.info(f"Vector size of Text Vocabulary: {text.vocab.vectors.shape[1]}")
    logging.info("Label Length: " + str(len(label.vocab)))


def _build_train_set(p_bunch: Bunch, u_bunch: Bunch, n_bunch: Optional[Bunch],
                     text: Field, label: LabelField) -> Dataset:
    r"""
    Convert the positive, negative, and unlabeled \p Bunch objects into a Dataset
    """
    data, labels, names = [], [], []
    for bunch, lbl in ((n_bunch, NEG_LABEL), (u_bunch, U_LABEL), (p_bunch, POS_LABEL)):
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
                 fix_length=args.seq_len)
    # noinspection PyPep8Naming
    LABEL = LabelField(sequential=False)
    complete_ds = _bunch_to_ds(complete_train, TEXT, LABEL)
    TEXT.build_vocab(complete_ds, min_freq=2,
                     vectors=torchtext.vocab.GloVe(name="6B", dim=args.embed_dim, cache=cache_dir))

    p_bunch, u_bunch = _select_bunch_uar(args.size_p, complete_train, p_cls,
                                         remove_from_bunch=False)
    n_bunch, u_bunch = _select_negative_bunch(args.size_n, u_bunch, n_cls, args.bias,
                                              remove_from_bunch=False)
    u_bunch = _reduce_to_fixed_size(u_bunch, new_size=args.size_u)

    test_bunch = _download_20newsgroups("test", data_dir, p_cls, n_cls)

    # Binarize the labels
    for bunch in (p_bunch, u_bunch, n_bunch, test_bunch):
        _configure_binary_labels(bunch, pos_cls=p_cls, neg_cls=n_cls)

    # Sanity check
    assert np.all(p_bunch[LABEL_COL] == POS_LABEL), "Negative example in positive (labeled) set"
    assert len(p_bunch[LABEL_COL]) == args.size_p, "Positive set has wrong number of examples"
    assert np.all(n_bunch[LABEL_COL] == NEG_LABEL), "Positive example in negative (labeled) set"
    assert len(n_bunch[LABEL_COL]) == args.size_n, "Negative set has wrong number of examples"
    assert len(u_bunch[LABEL_COL]) == args.size_u, "Unlabeled set has wrong number of examples"

    ng_data = NewsgroupsData(text=TEXT, label=LABEL)
    ng_data.train = _build_train_set(p_bunch, u_bunch, n_bunch, TEXT, LABEL)
    ng_data.unlabel = _bunch_to_ds(u_bunch, TEXT, LABEL)
    ng_data.test = _bunch_to_ds(test_bunch, TEXT, LABEL)

    tot_unlabel_size = args.size_p + args.size_n + args.size_u
    assert len(ng_data.train.examples) == tot_unlabel_size, "Train dataset is wrong size"

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
