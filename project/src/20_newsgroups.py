from fastai.basic_data import DataBunch


def merge_dbs_for_latent(p_db: DataBunch, bn_db: DataBunch, u_db: DataBunch) -> DataBunch:
    r"""
    Merge the three \p DataBunch objects.  Positive and biased negative will be positive labeled
    since they have :math:`s = 1` while the unlabeled \p DataBunch will be negative labeled.

    :return: Merged \p DataBunch for learning
    """

