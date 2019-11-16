from argparse import Namespace

from fastai.basic_data import DataBunch
from torch import Tensor
import torch.nn as nn
import torch.optim
from torchtext.data import Iterator

from pubn import IS_CUDA, TORCH_DEVICE



class NlpBiasedLearner(nn.Module):
    def __init__(self, embedding: nn.Embedding, args: Namespace):
        self._model = RnnClassifier(embedding)
        self.l_type = args.l_type()

    def fit(self, train: Iterator):
        if isinstance(self.l_type, PULoss):
            self._fit_nnpu(train)

    def _fit_supervised(self, train: Iterator):
        pass


    def _fit_

    def _train_sigma(self, p_db: DataBunch, bn_db: DataBunch, u_db: DataBunch):
        raise NotImplementedError

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
