import numpy as np

def generate_channel(Nt, theta_DoA, d, lamada):

    beta = np.exp(-1j * 2 * np.pi * np.random.rand())

    nnt = np.arange(0, Nt).reshape(-1, 1)

    codeword = np.arange(-1 + 1/Nt, 1, 2/Nt)

    h_bs = np.sqrt(1/Nt) * np.exp(
        1j * nnt * 2 * np.pi * d * codeword[int(theta_DoA)] / lamada
    )

    h = np.sqrt(Nt) * beta * h_bs.conj().T

    return h, h_bs
