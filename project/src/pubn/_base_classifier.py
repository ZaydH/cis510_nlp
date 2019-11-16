from torch import Tensor
import torch.nn as nn

from pubn import TORCH_DEVICE


class ClassifierConfig:
    BIDIRECTIONAL = True
    EMBED_DIM = 300

    RNN_HIDDEN_DIM = 300
    RNN_DEPTH = 1

    FF_HIDDEN_DEPTH = 1
    FF_HIDDEN_DIM = 256
    FF_ACTIVATION = nn.ReLU

    BASE_RNN = nn.LSTM


class BaseClassifier(nn.Module):
    Config = ClassifierConfig

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
