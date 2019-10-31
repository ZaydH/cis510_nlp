# -*- utf-8 -*-
r"""
    tensor_utils.py
    ~~~~~~~~~~~~~~~

    Defines the \p TensorUtils class, which provides a standard interface when using \p torch and
    \p numpy tensors.

    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""

from typing import Optional, Tuple, Union

import numpy as np

import torch
from torch import Tensor

from .custom_types import ListOrInt, OptInt, TorchOrNp

SizeType = Union[int, Tuple[int, ...]]


class TensorUtils:
    r""" Static class that makes a consistent interface for NumPy and torch tensors """
    @classmethod
    def concat(cls, tensors, dim: OptInt = None) -> TorchOrNp:
        r"""
        Concatenates the set of tensors

        :param tensors: Tensors to concatenate
        :param dim: Dimension along which to concatenate
        :return: Concatenated tensor
        """
        if cls.is_np(tensors[0]):
            return np.concatenate(tensors, axis=dim)
        if dim is None:
            dim = 0
        return torch.cat(tensors, dim=dim)

    @classmethod
    def ones(cls, like: TorchOrNp, size: SizeType) -> TorchOrNp:
        r""" Creates a ONES tensor with the same shape, type, and dtype as \p like """
        if cls.is_np(like):
            return np.ones(size)
        if isinstance(size, int):
            size = cls._convert_int_size(size)
        return torch.ones(size, dtype=like.dtype)

    @classmethod
    def zeros(cls, like: TorchOrNp, size: SizeType) -> TorchOrNp:
        r""" Creates a ZEROS tensor with the same shape, type, and dtype as \p like """
        if cls.is_np(like):
            return np.zeros(size)
        if isinstance(size, int):
            size = cls._convert_int_size(size)
        return torch.zeros(size, dtype=like.dtype)

    @classmethod
    def full(cls, like: TorchOrNp, size: SizeType, val) -> TorchOrNp:
        r""" Creates a prepopulated tensor with the same shape, type, and dtype as \p like """
        if cls.is_np(like):
            return np.full(size, val)
        if isinstance(size, int):
            size = cls._convert_int_size(size)
        return torch.full(size, val, dtype=like.dtype)

    @staticmethod
    def _convert_int_size(size: int) -> Tuple[int]:
        r""" Standardizes converting \p int objects in \p ones, \p zeros, and \p full methods """
        # noinspection PyRedundantParentheses
        return (size,)

    @classmethod
    def abs(cls, x: TorchOrNp) -> TorchOrNp:
        r""" Performs the absolute value on \p x """
        if cls.is_np(x):
            return np.abs(x)
        return x.abs()

    @classmethod
    def min(cls, x: TorchOrNp):
        r""" Finds the MINIMUM value in \p x """
        min_val = x.min()
        if cls.is_torch(x):
            min_val = min_val.item()
        return min_val

    @classmethod
    def max(cls, x: TorchOrNp):
        r""" Finds the MAXIMUM value in \p x """
        max_val = x.max()
        if cls.is_torch(x):
            max_val = max_val.item()
        return max_val

    @classmethod
    def split(cls, x: TorchOrNp, indices_or_sections: ListOrInt) -> TorchOrNp:
        r"""
        Divides a tensor into multiple pieces

        :param x: Tensor to split
        :param indices_or_sections: If an integer, N, the array will be divided
                                    into N equal arrays along `axis`.
                                    If such a split is not possible, an error is raised.

        If `indices_or_sections` is a 1-D array of sorted integers, the entries
        indicate where along `axis` the array is split.  For example,
        ``[2, 3]`` would, for ``axis=0``, result in

          - ary[:2]
          - ary[2:3]
          - ary[3:]

        If an index exceeds the dimension of the array along `axis`,
        an empty sub-array is returned correspondingly.
        """
        if cls.is_np(x):
            return np.split(x, indices_or_sections)
        return torch.split(x, indices_or_sections)

    @staticmethod
    def is_np(tensor: TorchOrNp) -> bool:
        r""" Return \p True if \p tensor is a \p numpy \p nd_array """
        return isinstance(tensor, np.ndarray)

    @staticmethod
    def is_torch(tensor: TorchOrNp) -> bool:
        r""" Return \p True if \p tensor is a \p torch tensor """
        return isinstance(tensor, Tensor)

    @classmethod
    def to_numpy(cls, tensor: TorchOrNp, dtype=None) -> np.ndarray:
        r""" Return a numpy tensor """
        if cls.is_torch(tensor):
            tensor = tensor.detach().cpu().numpy()
        if dtype is not None:
            tensor = tensor.astype(dtype)
        return tensor

    @classmethod
    def to_torch(cls, tensor: TorchOrNp, dtype: Optional[torch.dtype] = None) -> Tensor:
        r""" Return a numpy tensor """
        if cls.is_np(tensor):
            tensor = torch.from_numpy(tensor)
        if dtype is not None:
            tensor = tensor.type(dtype)
        return tensor

    @classmethod
    def flatten(cls, x: TorchOrNp) -> TorchOrNp:
        r""" Flattens vector \p x to be a single dimension """
        is_np = cls.is_np(x)
        if is_np:
            x = torch.from_numpy(x)
        x = x.view([x.shape[0], -1])
        if is_np:
            x = x.cpu().numpy()
        return x
