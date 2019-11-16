import logging
from argparse import Namespace

from fastai.basic_data import DataBunch
from torch import Tensor
import torch.nn as nn
import torch.optim
from torchtext.data import Iterator

from pubn.logger import TrainingLogger, create_stdout_handler
from ._base_classifier import BaseClassifier, ClassifierConfig
from .config import IS_CUDA, TORCH_DEVICE
from .loss import LossType, PULoss, PUbN


class NlpBiasedLearner(nn.Module):
    Config = ClassifierConfig

    _log = None

    # def __init__(self, embedding: nn.Embedding, args: Namespace):
    def __init__(self, args: Namespace, embedding: nn.Embedding):
        self._setup_logger()
        super().__init__()

        self._model = BaseClassifier(embedding)
        self.l_type = args.loss

    def fit(self, train: Iterator):
        if self.l_type == LossType.NNPU:
            self._fit_nnpu(train)
        elif self.l_type == LossType.PUBN:
            self._fit_pubn(train)
        elif self.l_type == LossType.PN:
            self._fit_supervised(train)
        else:
            raise ValueError("Unknown loss type")

    def _fit_nnpu(self, train: Iterator):
        r""" Use the nnPU loss """
        # noinspection PyUnresolvedReferences
        for ep in range(1, self._model.Config.NUM_EPOCH + 1):
            for batch in train:
                pass

    def _fit_pubn(self, train: Iterator):
        raise NotImplementedError("_fit_pubn not implemented")

    def _train_sigma(self, p_db: DataBunch, bn_db: DataBunch, u_db: DataBunch):
        raise NotImplementedError("sigma trainer not implemented")

    def _fit_supervised(self, train: Iterator):
        pass

    def predict(self, test: Iterator):
        raise NotImplementedError("predict function not implemented")

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
