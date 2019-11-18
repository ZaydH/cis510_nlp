from argparse import Namespace
import logging
from functools import partial
from pathlib import Path
import time
from typing import Callable

import numpy as np

from fastai.basic_data import DataBunch
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Iterator, Dataset, LabelField

from ._base_classifier import BaseClassifier, ClassifierConfig
from ._utils import BASE_DIR, NEG_LABEL, POS_LABEL, TORCH_DEVICE, U_LABEL, construct_iterator, \
    construct_filename
from .logger import TrainingLogger, create_stdout_handler
from .loss import LossType, PULoss


class NlpBiasedLearner(nn.Module):
    Config = ClassifierConfig

    _log = None

    # def __init__(self, embedding: nn.Embedding, args: Namespace):
    def __init__(self, args: Namespace, embedding_weights: Tensor, prior: float):
        super().__init__()
        self._setup_logger()

        self._log.debug(f"NLP Learner: Prior: {prior:.3f}")

        self._model = BaseClassifier(embedding_weights)

        self._args = args
        self.prior = prior
        self.l_type = args.loss

        self._prefix = self.l_type.name.lower()
        self._train_start = self._best_loss = self._logger = None

    def fit(self, train: Dataset, labels: LabelField):
        r""" Fits the learner"""
        # Fields that apply regardless of loss method
        self._train_start = time.time()
        self._best_loss = np.inf

        tb_dir = BASE_DIR / "tb"
        TrainingLogger.create_tensorboard(tb_dir)
        self._logger = TrainingLogger(["Train Loss", "Valid Loss", "Best", "Time"],
                                      [20, 20, 7, 10],
                                      logger_name=self.Config.LOGGER_NAME,
                                      tb_grp_name=self.l_type.name)

        # Filter the dataset based on the training configuration
        if self.l_type == LossType.NNPU:
            train = exclude_label_in_dataset(train, NEG_LABEL)
        elif self.l_type == LossType.PN:
            train = exclude_label_in_dataset(train, U_LABEL)
        # noinspection PyUnresolvedReferences
        itr = construct_iterator(train, bs=self._model.Config.BATCH_SIZE, shuffle=True)

        if self.l_type == LossType.NNPU:
            self._fit_nnpu(itr, labels)
        elif self.l_type == LossType.PUBN:
            self._fit_pubn(itr)
        elif self.l_type == LossType.PN:
            self._fit_supervised(itr)
        else:
            raise ValueError("Unknown loss type")

    def _fit_nnpu(self, train: Iterator, labels: LabelField):
        r""" Use the nnPU loss """
        pu_loss = PULoss(prior=self.prior, pos_label=labels.vocab.stoi[POS_LABEL])

        # noinspection PyUnresolvedReferences
        opt = optim.Adam(self.parameters(), lr=self._model.Config.LEARNING_RATE,
                         weight_decay=self._model.Config.WEIGHT_DECAY)
        # noinspection PyUnresolvedReferences
        for ep in range(1, self._model.Config.NUM_EPOCH + 1):
            train_loss, num_batch = torch.zeros(()), 0
            for batch in train:
                opt.zero_grad()
                # noinspection PyUnresolvedReferences
                dec_scores = self._model.forward(*batch.text)

                loss = pu_loss.calc_loss(dec_scores=dec_scores, label=batch.label)
                train_loss += loss.loss_var.detach()
                num_batch += 1

                loss.grad_var.backward()
                opt.step()

            train_loss /= num_batch
            valid_loss = self._calc_valid_loss(train, partial(pu_loss.zero_one_loss))
            self._log_epoch(ep, train_loss, valid_loss)
        self._restore_best_model()

    def _fit_pubn(self, train: Iterator):
        raise NotImplementedError("_fit_pubn not implemented")

    def _train_sigma(self, p_db: DataBunch, bn_db: DataBunch, u_db: DataBunch):
        raise NotImplementedError("sigma trainer not implemented")

    def _fit_supervised(self, train: Iterator):
        pass

    def _log_epoch(self, ep: int, train_loss: Tensor, valid_loss: Tensor):
        r"""
        Log the results of the epoch

        :param ep: Epoch number
        :param train_loss: Training loss value
        :param valid_loss: Validation loss value
        """
        is_best = float(valid_loss.item()) < self._best_loss
        if is_best:
            self._best_loss = float(valid_loss.item())
            save_module(self, self._build_serialize_name(self._prefix))
        self._logger.log(ep, [train_loss, valid_loss, is_best, time.time() - self._train_start])

    def _calc_valid_loss(self, itr: Iterator, loss_func: Callable):
        r""" Calculate the validation loss for \p itr """
        self.eval()

        tot_loss, n_batch = torch.zeros(()), 0
        with torch.no_grad():
            for batch in itr:
                n_batch += 1
                tot_loss += loss_func(self.forward(*batch.text), batch.label).detach()

        return tot_loss / n_batch

    def forward(self, x: Tensor, x_len: Tensor) -> Tensor:
        # noinspection PyUnresolvedReferences
        return self._model.forward(x, x_len)

    def _restore_best_model(self):
        r""" Restores the best trained model from disk """
        msg = f"Restoring {self.l_type.name} best trained model"
        self._log.debug(f"Starting: {msg}")
        load_module(self, self._build_serialize_name(self._prefix))
        self._log.debug(f"COMPLETED: {msg}")

    def _build_serialize_name(self, prefix: str) -> Path:
        r"""

        :param prefix: Prefix given to the name of the serialized file
        :return: \p Path to the serialized file
        """
        serialize_dir = BASE_DIR / "models"
        serialize_dir.mkdir(parents=True, exist_ok=True)
        return construct_filename(prefix, self._args, serialize_dir, "pth")

    @classmethod
    def _setup_logger(cls) -> None:
        r""" Creates a logger for just the ddPU class """
        if cls._log is not None: return
        cls._log = logging.getLogger(cls.Config.LOGGER_NAME)
        cls._log.propagate = False  # Do not propagate log messages to a parent logger
        create_stdout_handler(cls.Config.LOG_LEVEL, logger_name=cls.Config.LOGGER_NAME)
#
#
# class _SigmaLearner(nn.Module):
#     class Config:
#         NUM_EPOCH = 100
#         LR = 1E-3
#         WD = 0
#
#     def __init__(self):
#         self._net = dfdf
#
#         if IS_CUDA: self.cuda(TORCH_DEVICE)
#
#     def fit(self, p_db: DataBunch, bn_db: DataBunch, u_db: DataBunch):
#         merged_db = merge_dbs_for_latent(p_db=p_db, bn_db=bn_db, u_db=u_db, neg_label=PUbN.BN_LABEL)
#
#
#         opt = torch.optim.AdamW(self.parameters(), lr=self.Config.LR, weight_decay=self.Config.WD)
#         for ep in range(1, self.Config.NUM_EPOCH + 1):
#
#             for x, y in merged_db.train_dl:
#                 dec_scores = self._net.forward(x)
#
#
#     def forward(self, x: Tensor) -> Tensor:
#         return self._net.forward(x)


def save_module(module: nn.Module, filepath: Path) -> None:
    r""" Save the specified \p model to disk """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # noinspection PyUnresolvedReferences
    torch.save(module.state_dict(), str(filepath))


def load_module(module: nn.Module, filepath: Path):
    r"""
    Loads the specified model in file \p filepath into \p module and then returns \p module.

    :param module: \p Module where the module on disk will be loaded
    :param filepath: File where the \p Module is stored
    :return: Loaded model
    """
    # Map location allows for mapping model trained on any device to be loaded
    module.load_state_dict(torch.load(str(filepath), map_location=TORCH_DEVICE))
    # module.load_state_dict(torch.load(str(filepath)))

    module.eval()
    return module


def exclude_label_in_dataset(ds: Dataset, label_to_exclude: int) -> Dataset:
    r""" Creates a new \p Dataset that will exclude the label of \p label_to_exclude """
    return Dataset(examples=ds.examples, fields=ds.fields,
                   filter_pred=lambda x: x != label_to_exclude)
