import numpy as np

def generate_widebeam(Nt, index):

    nn = np.arange(0, Nt).reshape(-1, 1)

    codeword = np.arange(-1 + 1/Nt, 1, 2/Nt)

    codebook = np.exp(1j * np.pi * nn @ codeword.reshape(1, -1)) / np.sqrt(Nt)

    v = np.zeros((Nt, 1), dtype=complex)

    index_length = len(index)

    for idx in range(index_length):

        k = int(index[idx])

        v += np.exp(1j * np.pi * k * (-1 + 1/Nt)) * codebook[:, k].reshape(-1,1)

    v = v / np.linalg.norm(v)

    return v
