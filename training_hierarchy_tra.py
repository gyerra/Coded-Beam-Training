import numpy as np

def training_hierarchy_tra(w_hierarchy_tra, h, SNR, Nt):

    l0 = int(np.log2(Nt))

    w_BS = np.zeros((Nt, 2), dtype=complex)

    sigma = np.sqrt(1 / SNR)

    nnt = np.arange(Nt).reshape(-1, 1)

    codeword_BS = np.linspace(-1 + 1/Nt, 1 - 1/Nt, Nt)

    codebook_BS = np.exp(1j * np.pi * nnt @ codeword_BS.reshape(1, -1)) / np.sqrt(Nt)

    i = 0   # Python uses 0-indexing

    for l in range(l0):

        w_BS[:, 0] = w_hierarchy_tra[l][i]
        w_BS[:, 1] = w_hierarchy_tra[l][i + 1]

        noise = sigma * (np.random.randn(2) + 1j*np.random.randn(2)) / np.sqrt(2)

        temp = h.T @ w_BS + noise

        A = np.abs(temp) ** 2

        gain = np.max(A)

        if l < l0 - 1:

            if A[0] > A[1]:
                i = i * 2
            else:
                i = (i + 1) * 2

        else:
            if A[0] < A[1]:
                i = i + 1

    id_BS = i

    array_gain = np.abs(h.T @ codebook_BS[:, id_BS]) ** 2

    return array_gain, id_BS
