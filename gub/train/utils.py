import torch
import numpy as np
import time
import os


def feat_preprocess(features, feat_norm=None, device="cpu"):
    r"""

    Description
    -----------
    Preprocess the features.

    Parameters
    ----------
    features : torch.Tensor or numpy.array
        Features in form of torch tensor or numpy array.
    feat_norm : str, optional
        Type of features normalization, choose from ["arctan", "tanh", None]. Default: ``None``.
    device : str, optional
        Device used to host data. Default: ``cpu``.

    Returns
    -------
    features : torch.Tensor
        Features in form of torch tensor on chosen device.

    """

    def feat_normalize(feat, norm=None):
        if norm == "arctan":
            feat = 2 * np.arctan(feat) / np.pi
        elif norm == "tanh":
            feat = np.tanh(feat)
        else:
            feat = feat

        return feat

    if type(features) != torch.Tensor:
        features = torch.FloatTensor(features)
    elif features.type() != "torch.FloatTensor":
        features = features.float()
    if feat_norm is not None:
        features = feat_normalize(features, norm=feat_norm)

    features = features.to(device)

    return features


def label_preprocess(labels, device="cpu"):
    r"""

    Description
    -----------
    Convert labels to torch tensor.

    Parameters
    ----------
    labels : torch.Tensor
        Labels in form of torch tensor.
    device : str, optional
        Device used to host data. Default: ``cpu``.

    Returns
    -------
    labels : torch.Tensor
        Features in form of torch tensor on chosen device.

    """

    if type(labels) != torch.Tensor:
        labels = torch.LongTensor(labels)
    elif labels.type() != "torch.LongTensor":
        labels = labels.long()

    labels = labels.to(device)

    return labels


def save_model(model, save_dir, name, verbose=True):
    r"""

    Description
    -----------
    Save trained model.

    Parameters
    ----------
    model : torch.nn.module
        Model implemented based on ``torch.nn.module``.
    save_dir : str
        Directory to save the model.
    name : str
        Name of saved model.
    verbose : bool, optional
        Whether to display logs. Default: ``False``.

    """

    if save_dir is None:
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        save_dir = "./tmp_{}".format(cur_time)
        os.makedirs(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model, os.path.join(save_dir, name))

    if verbose:
        print("Model saved in '{}'.".format(os.path.join(save_dir, name)))
