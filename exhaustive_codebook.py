import numpy as np

def exhaustive_codebook(N):

    nn = np.arange(0, N).reshape(-1, 1)

    codeword = np.arange(-1 + 1/N, 1, 2/N)

    w_far = np.exp(1j * np.pi * nn @ codeword.reshape(1, -1)) / np.sqrt(N)

    return w_far
