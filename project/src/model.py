from argparse import Namespace

from fastai.basic_data import DataBunch
from torch import Tensor
import torch.nn as nn
import torch.optim

from newsgroups import merge_dbs_for_latent
from pu_loss import PULoss, PUbN

IS_CUDA = torch.cuda.is_available()



class NlpBiasedLearner(nn.Module):
    def __init__(self, args: Namespace):

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
