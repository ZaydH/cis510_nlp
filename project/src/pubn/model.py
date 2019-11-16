from argparse import Namespace

from fastai.basic_data import DataBunch
from torch import Tensor
import torch.nn as nn
import torch.optim
from torchtext.data import Iterator

from ._base_classifier import BaseClassifier
from .config import IS_CUDA, TORCH_DEVICE
from .pu_loss import PUbN, PULoss


class NlpBiasedLearner(nn.Module):
    # def __init__(self, embedding: nn.Embedding, args: Namespace):
    def __init__(self, args: Namespace, embedding: nn.Embedding):
        self._model = BaseClassifier(embedding)
        self.l_type = args.l_type.value()

        self._log =

    def fit(self, train: Iterator):
        if isinstance(self.l_type, PULoss):
            self._fit_nnpu(train)
        elif isinstance(self.l_type, PUbN):
            self._fit_pubn(train)
        else:
            self._fit_supervised(train)

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


class _SigmaLearner(nn.Module):
    class Config:
        NUM_EPOCH = 100
        LR = 1E-3
        WD = 0

    def __init__(self):
        self._net = dfdf

        if IS_CUDA: self.cuda(TORCH_DEVICE)

    def fit(self, p_db: DataBunch, bn_db: DataBunch, u_db: DataBunch):
        merged_db = merge_dbs_for_latent(p_db=p_db, bn_db=bn_db, u_db=u_db, neg_label=PUbN.BN_LABEL)


        opt = torch.optim.AdamW(self.parameters(), lr=self.Config.LR, weight_decay=self.Config.WD)
        for ep in range(1, self.Config.NUM_EPOCH + 1):

            for x, y in merged_db.train_dl:
                dec_scores = self._net.forward(x)


    def forward(self, x: Tensor) -> Tensor:
        return self._net.forward(x)
