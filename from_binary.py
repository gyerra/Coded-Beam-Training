import numpy as np

def from_binary(bin_form):
    """
    Convert a binary vector (MSB first, LSB last) to an integer.
    """

    L = len(bin_form)

    nv = 2 ** np.arange(L-1, -1, -1)

    ret = np.dot(bin_form, nv)

    return int(ret)
