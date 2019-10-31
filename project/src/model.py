from enum import Enum

from fastai.basic_data import DataBunch
import torch.nn as nn


class LossType(Enum):
    r""" Loss type to train the learner """
    PN = "PU"
    NNPU = "nnPU"
    PUBN = "PUbN"


class NlpBiasedLearner(nn.Module):


    def _train_sigma(self, p_db: DataBunch, bn_db: DataBunch, u_db: DataBunch):

