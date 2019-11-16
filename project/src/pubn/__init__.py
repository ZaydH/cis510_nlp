from ._base_classifier import ClassifierConfig

from ._utils import BASE_DIR, IS_CUDA, NEG_LABEL, POS_LABEL, U_LABEL, construct_iterator


def calculate_prior(ds) -> float:
    r""" Calculates the positive prior probability over the labels set """
    labels = [x.label for x in ds]
    assert all(x != U_LABEL for x in labels), "Unlabeled elements not allowed in prior calculation"
    num_pos = sum([1 if x == POS_LABEL else 0 for x in labels])
    assert num_pos > 0, "No positive elements found"
    return num_pos / len(labels)
