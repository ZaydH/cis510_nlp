import collections
from enum import Enum
from typing import Callable

import torch
from torch import Tensor
import torch.nn as nn


def _default_loss(x: Tensor) -> Tensor:
    r""" Default loss function for the loss functions """
    return torch.sigmoid(-x)


PU_SIGMOID_LOSS = _default_loss


class PULoss:
    """wrapper of loss function for PU learning"""

    LossInfo = collections.namedtuple("LossInfo", ["loss_var", "grad_var"])

    def __init__(self, prior: float, pos_label: int, loss: Callable = _default_loss,
                 gamma: float = 1, beta: float = 0, use_nnpu: bool = True):
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.pos_label = pos_label

        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss
        self._is_nnpu = use_nnpu

    @property
    def is_nnpu(self) -> bool:
        r"""
        Returns \p True if the loss is the nnPU loss.  Otherwise if it is the uPU loss, it returns
        \p False.
        """
        return self._is_nnpu

    def name(self) -> str:
        r""" Name of the loss, either "nnPU" or "uPU" """
        return "nnPU" if self.is_nnpu else "uPU"

    @staticmethod
    def _is_torch(tensor: Tensor) -> bool:
        r""" Return \p True if \p tensor is a \p torch tensor """
        return isinstance(tensor, Tensor)

    @classmethod
    def _verify_loss_inputs(cls, dec_scores: Tensor, labels: Tensor) -> None:
        r""" Sanity check the inputs """
        assert cls._is_torch(dec_scores), "dec_scores must be torch tensor"
        assert cls._is_torch(labels), "labels must be torch tensor"

        assert len(dec_scores.shape) == 1, "dec_scores should be a vector"
        assert len(labels.shape) == 1, "labels should be a vector"
        assert dec_scores.shape[0] == labels.shape[0], "Batch size mismatch"

        assert dec_scores.dtype == torch.float, "dec_scores tensor must be float"
        assert labels.dtype == torch.int64, "labels must be integers"

    def calc_loss(self, dec_scores: Tensor, label: Tensor) -> 'LossInfo':
        r"""
        nnPU uses separate approaches for determining the loss and variable used for calculating
        the gradient.
        :param dec_scores: Decision function value
        :param label: Labels for each sample in \p.
        :return: Named tuple with value used for loss and the one used for the gradient
        """
        dec_scores, label = dec_scores.squeeze(), label.squeeze()
        self._verify_loss_inputs(dec_scores, label)

        # Mask used to filter the dec_scores tensor and in loss calculations
        p_mask = label == self.pos_label
        u_mask = ~p_mask
        has_p, has_u = p_mask.any(), u_mask.any()

        y_unlabel = self.loss_func(-dec_scores)
        neg_risk = y_unlabel[u_mask].mean() if has_u else torch.zeros(())
        if has_p:
            y_pos = self.loss_func(dec_scores[p_mask])
            pos_risk = self.prior * y_pos.mean()

            neg_risk -= self.prior * y_unlabel[p_mask].mean()
        else:
            pos_risk = torch.zeros(())  # Needs to be have len(shape) == 0

        loss = gradient_var = pos_risk + neg_risk
        if self.is_nnpu and neg_risk < -self.beta:
            loss = pos_risk - self.beta
            gradient_var = -self.gamma * neg_risk
        return self.LossInfo(loss_var=loss, grad_var=gradient_var)

    def zero_one_loss(self, dec_scores: Tensor, labels: Tensor) -> Tensor:
        r"""
        In validation, 0/1 loss is used for validation. This method implements that approach.

        :param dec_scores: Decision function value
        :param labels: Labels for each sample in \p.
        :return: Named tuple with value used for loss and the one used for the gradient
        """
        dec_scores, labels = dec_scores.squeeze(), labels.squeeze()
        self._verify_loss_inputs(dec_scores, labels)

        # Mask used to filter the dec_scores tensor and in loss calculations
        p_mask = labels == self.pos_label
        one_val = 1  # Sign value for examples to be positively labeled
        p_err = (dec_scores[p_mask].sign() != one_val).float().mean()

        u_mask = ~p_mask
        # By checking against P_LABEL, inherent negation so do not use != operator
        u_err = (dec_scores[u_mask].sign() == one_val).float().mean()

        return 2 * self.prior * p_err + u_err - self.prior


class PUbN:
    r"""
    Positive, unlabeled, and biased negative risk estimator from:

    Yu-Guan Hsieh, Gang Niu, and Masashi Sugiyama. Classification from Positive, Unlabeled and
    Biased Negative Data. ICML 2019.
    """
    POS_LABEL = 1
    BN_LABEL = -1

    def __init__(self, prior: float, rho: float, eta: float, sigma: nn.Module,
                 loss: Callable = _default_loss):
        self.prior = prior
        self.rho = rho
        self.eta = eta

        self.sigma = sigma
        self.loss_func = loss

    def calc_loss(self, x: Tensor, dec_scores: Tensor, labels: Tensor) -> Tensor:
        # Labeled loss terms
        p_mask, bn_mask = (labels == self.POS_LABEL), (labels == self.BN_LABEL)
        u_mask = p_mask.logical_xor(bn_mask).logical_not()

        l_pos = self.prior * self.loss_func(dec_scores[p_mask]) if p_mask.any() else torch.zeros(())
        l_bn = self.rho * self.loss_func(dec_scores[bn_mask]) if bn_mask.any() else torch.zeros(())

        self.sigma.eval()
        with torch.no_grad():
            sigma_x = self.sigma(x)
        l_u_n = self._unlabeled_neg_loss(u_mask, sigma_x, dec_scores, is_unlabeled=True) \
                + self._unlabeled_neg_loss(p_mask, sigma_x, dec_scores, is_unlabeled=False) \
                + self._unlabeled_neg_loss(bn_mask, sigma_x, dec_scores, is_unlabeled=False)

        return l_u_n + self.prior * l_pos + self.rho * l_bn

    def _unlabeled_neg_loss(self, mask: Tensor, sigma_x: Tensor, dec_scores: Tensor,
                            is_unlabeled: bool) -> Tensor:
        r"""
        Calculates a single term of the expected
        :param mask: Masks for either
        :param sigma_x: Estimated value of :math:`\sigma(x)`.
        :param dec_scores: Decision function scores
        :param is_unlabeled: If \p True, considering the negative set
        :return: Single term in the unlabeled negative set
        """
        # First time through the loop applies mask generally. Second pass filters based on
        # eta
        for i in range(2):
            sigma_x, dec_scores = sigma_x[mask], dec_scores[mask]

            if sigma_x.numel() == 0: return torch.zeros(())
            mask = sigma_x > self.eta
            if is_unlabeled: mask = ~mask

        loss = self.loss_func(-dec_scores)
        neg_sigma = -sigma_x + 1
        if not is_unlabeled: neg_sigma = neg_sigma / sigma_x

        return (loss * neg_sigma).mean()


class LossType(Enum):
    r""" Loss type to train the learner """
    PN = nn.BCELoss
    NNPU = PULoss
    PUBN = PUbN
