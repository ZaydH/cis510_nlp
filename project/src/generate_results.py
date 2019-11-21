from argparse import Namespace
from dataclasses import dataclass
import logging

import numpy as np
from sklearn.metrics import confusion_matrix, average_precision_score, f1_score

from fastai.metrics import auc_roc_score
import torch
from torchtext.data import Dataset, LabelField

from pubn import BASE_DIR, POS_LABEL, construct_iterator
from pubn.model import NlpBiasedLearner


@dataclass
class LearnerResults:
    @dataclass(init=True)
    class DatasetResult:
        ds_size: int
        accuracy: float = None
        auroc: float = None
        auprc: float = None
        f1: float = None

    valid_loss = None
    unlabel = None
    test = None


def calculate_results(args: Namespace, classifier: NlpBiasedLearner, labels: LabelField,
                      unlabel_ds: Dataset, test_ds: Dataset):
    classifier.eval()

    res = LearnerResults()
    res.valid_loss = classifier.best_loss

    for ds, name in ((unlabel_ds, "unlabel"), (test_ds, "test")):
        itr = construct_iterator(ds, bs=args.bs, shuffle=False)
        all_y, dec_scores = [], []
        with torch.no_grad():
            for batch in itr:
                all_y.append(batch.label)
                dec_scores.append(classifier.forward(*batch.text))

        # Iterator transforms label so transform it back
        tfm_y = torch.cat(all_y, dim=0).squeeze()
        y = torch.full_like(tfm_y, -1)
        y[tfm_y == labels.vocab.stoi[POS_LABEL]] = 1
        y = y.cpu().numpy()

        dec_scores = torch.cat(dec_scores, dim=0).squeeze()
        y_hat, dec_scores = dec_scores.sign().cpu().numpy(), dec_scores.cpu().numpy()

        res.__setattr__(name, _single_ds_results(name, args, y, y_hat, dec_scores))


def _single_ds_results(ds_name: str, args: Namespace, y: np.ndarray, y_hat: np.ndarray,
                       dec_scores: np.ndarray) -> LearnerResults.DatasetResult:
    loss_name = args.loss.name
    results = LearnerResults.DatasetResult(y.shape[0])

    str_prefix = f"{loss_name} {ds_name}:"

    logging.debug(f"{str_prefix} Dataset Size: %d", results.ds_size)
    # Pre-calculate fields needed in other calculations
    results.conf_matrix = confusion_matrix(y, y_hat)
    assert np.sum(results.conf_matrix) == results.ds_size, "Verify size matches"

    # Calculate prior information
    results.accuracy = np.trace(results.conf_matrix) / results.ds_size
    logging.debug(f"{str_prefix} Accuracy = %.3f%%", 100. * results.accuracy)

    results.auroc = auc_roc_score(torch.tensor(dec_scores).cpu(), torch.tensor(y).cpu())
    logging.debug(f"{str_prefix} AUROC: %.6f", results.auroc)

    results.auprc = average_precision_score(y, dec_scores)
    logging.debug(f"{str_prefix} AUPRC %.6f", results.auprc)

    results.f1 = float(f1_score(y, y_hat))
    logging.debug(f"{str_prefix} F1-Score: %.6f", results.f1)

    logging.debug(f"{str_prefix} Confusion Matrix:\n{results.conf_matrix}")

    return results


def _write_results_to_disk(args: Namespace) -> None:
    results_dir = BASE_DIR / "results"
