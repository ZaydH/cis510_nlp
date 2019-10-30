from pathlib import Path
from typing import Union

from _prob_struct import LabeledCorpus, UnlabeledCorpus


def _base_import(path: Path, is_labeled: bool) -> Union[LabeledCorpus, UnlabeledCorpus]:
    with open(str(path), "r") as f_in:
        lines = f_in.readlines()

    sentences, is_new = [], True
    for line in lines:
        if is_new:
            sentences.append([])
            is_new = False
        # Blank line is end of sentence
        if not line:
            is_new = True
            continue

        if is_labeled:
            split = line.split("\t")
            assert len(split) == 2, f"File line \"{line}\" appears invalid"
            line = tuple(split)
        sentences[-1].append(line)
    return sentences


def import_labeled_corpus(path: Path) -> LabeledCorpus:
    r"""
    Parses the input file and does minimum check for validity.
    :return: List of sentences.  Each sentence is a list of tuples where each tuple is the word
             and the corresponding label.
    """
    return _base_import(path, is_labeled=True)


def import_test_corpus(path: Path) -> UnlabeledCorpus:
    r"""
    Imports a test corpus file

    :param path: Path to the file containing the test set
    :return: \p List of sentences. Each sentence is a \p List of words.
    """
    return _base_import(path, is_labeled=True)
