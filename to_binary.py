import numpy as np

def to_binary(ind, bin_length):
    # returns binary form of integer ind (MSB first, LSB last)

    ret = np.zeros(bin_length, dtype=bool)

    for k in range(1, bin_length + 1):
        ret[bin_length - k] = ind % 2
        ind = ind // 2

    return ret
