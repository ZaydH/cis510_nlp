from typing import Optional

from fastai.basic_data import DataBunch
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset


def duplicate_no_drop_last_dataloader(dl: DataLoader, num_workers: int = None,
                                      shuffle: bool = False,
                                      pin_memory: Optional[bool] = None) -> DataLoader:
    r"""
    Helper function used for creating a duplicate \p DataLoader with \p drop_last disabled. This is
    useful to ensure that all data is exported as expected.
    """
    if num_workers is None:
        num_workers = dl.num_workers
    if pin_memory is None:
        pin_memory = dl.pin_memory
    return DataLoader(dl.dataset, shuffle=shuffle, drop_last=False, batch_size=dl.batch_size,
                      num_workers=num_workers, pin_memory=pin_memory)


def merge_dbs_for_latent(p_db: DataBunch, bn_db: DataBunch, u_db: DataBunch,
                         neg_label: int) -> DataBunch:
    r"""
    Merge the three \p DataBunch objects.  Positive and biased negative will be positive labeled
    since they have :math:`s = 1` while the unlabeled \p DataBunch will be negative labeled.

    :return: Merged \p DataBunch for learning
    """

    def _bn_like(_lbl: Tensor) -> Tensor:
        r""" Helper function used for  """
        return torch.full_like(_lbl, neg_label)

    all_x, all_y = [], []
    for db, y_func in ((p_db, torch.ones_like), (bn_db, _bn_like), (u_db, torch.zeros_like)):
        for _x, _y in duplicate_no_drop_last_dataloader(db.train_dl):
            all_x.append(_x)
            all_y.append(y_func(_y))

    x, y = torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)
    return DataBunch.create(TensorDataset(x, y), valid_ds=Dataset(), device=p_db.device,
                            num_workers=p_db.num_workers, bs=p_db.batch_size,
                            pin_memory=p_db.train_dl.pin_memory)
