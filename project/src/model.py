from argparse import Namespace

from fastai.basic_data import DataBunch
from torch import Tensor
import torch.nn as nn
import torch.optim

from newsgroups import merge_dbs_for_latent
from pu_loss import PULoss, PUbN

IS_CUDA = torch.cuda.is_available()
TORCH_DEVICE = torch.device("cuda:0" if IS_CUDA else "cpu")


class RnnClassifier(nn.Module):
    class Config:
        BIDIRECTIONAL = True
        EMBED_DIM = 300

        RNN_HIDDEN_DIM = 300
        RNN_DEPTH = 1

        FF_HIDDEN_DEPTH = 1
        FF_HIDDEN_DIM = 256
        FF_ACTIVATION = nn.ReLU

        BASE_RNN = nn.LSTM

    def __init__(self, embed: nn.Embedding):
        super().__init__()
        self._embed = embed

        self._rnn = self.Config.BASE_RNN(num_layers=self.Config.RNN_DEPTH,
                                         hidden_size=self.Config.RNN_HIDDEN_DIM,
                                         input_size=self.Config.EMBED_DIM,
                                         bidirectional=self.Config.BIDIRECTIONAL)

        self._ff = nn.Sequential()
        in_dim = self.Config.RNN_HIDDEN_DIM << (1 if self.Config.BIDIRECTIONAL else 0)
        for i in range(1, self.Config.FF_HIDDEN_DEPTH + 1):
            self._ff.add_module("Hidden_{i:02}_Lin", nn.Linear(in_dim, self.Config.FF_HIDDEN_DIM))
            self._ff.add_module("Hidden_{i:02}_Act", self.Config.FF_ACTIVATION())
            self._ff.add_module("Hidden_{i:02}_BN", nn.BatchNorm1d(self.Config.FF_HIDDEN_DIM))
            in_dim = self.Config.FF_HIDDEN_DIM
        # Add output layer
        self._ff.add_module("FF_Out", nn.Linear(in_dim, 1))
        # self._ff.add_module("FF_Out_Act", nn.Sigmoid())

        self.to(TORCH_DEVICE)

    def forward(self, x: Tensor) -> Tensor:
        x_embed = self._embed(x)

        # **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
        seq_out, _ = self._rnn.forward(x_embed, hx=None)  # Always use a fresh hidden

        ff_in = seq_out[-1]
        y_hat = self._ff.forward(ff_in)
        return y_hat


class NlpBiasedLearner(nn.Module):
    def __init__(self, embedding: nn.Embedding, args: Namespace):
        self._model = RnnClassifier(embedding)
        self.l_type = args.

    def fit(self):
        pass

    def _fit_supervised(self):
        pass

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
