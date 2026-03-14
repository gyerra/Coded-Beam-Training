import numpy as np

def generate_channel_multipath(Nt, theta_DoA, d, lamada):

    beta = np.exp(-1j * 2 * np.pi * np.random.rand())

    beta1 = 0.1 * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    beta2 = 0.1 * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

    nnt = np.arange(0, Nt).reshape(-1, 1)

    codeword = np.arange(-1 + 1/Nt, 1, 2/Nt)

    h_bs = np.sqrt(1/Nt) * (
        np.exp(1j * nnt * 2 * np.pi * d * codeword[int(theta_DoA[0])] / lamada)
        + beta1 * np.exp(1j * nnt * 2 * np.pi * d * codeword[int(theta_DoA[1])] / lamada)
        + beta2 * np.exp(1j * nnt * 2 * np.pi * d * codeword[int(theta_DoA[2])] / lamada)
    )

    h = np.sqrt(Nt) * beta * h_bs.conj().T

    return h, h_bs
