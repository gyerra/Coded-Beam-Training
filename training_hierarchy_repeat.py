import numpy as np

def training_hierarchy_repeat(w_hierarchy_far, Nt, H, K, SNR):

    l1 = int(np.log2(Nt))

    array_gain = np.zeros(K)
    id_BS = np.zeros(K, dtype=int)

    w_BS = np.zeros((Nt, 2), dtype=complex)

    sigma = np.sqrt(1 / SNR)

    nnt = np.arange(Nt).reshape(-1, 1)

    codeword_BS = np.linspace(-1 + 1/Nt, 1 - 1/Nt, Nt)

    codebook_BS = np.exp(1j * np.pi * nnt @ codeword_BS.reshape(1, -1)) / np.sqrt(Nt)

    # hierarchical training
    for l in range(l1 - 1):

        w_BS[:, 0] = w_hierarchy_far[l, 0, :]
        w_BS[:, 1] = w_hierarchy_far[l, 1, :]

        for k in range(K):

            noise = sigma * (np.random.randn(2) + 1j*np.random.randn(2)) / np.sqrt(2)

            temp = H[:, k].T @ w_BS + noise

            A = np.abs(temp) ** 2

            if A[0] >= A[1]:
                id_BS[k] = id_BS[k] * 2
            else:
                id_BS[k] = id_BS[k] * 2 + 1

    # final beam refinement
    for k in range(K):

        id1 = id_BS[k] * 2
        id2 = id_BS[k] * 2 + 1

        noise = sigma * (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)

        temp1 = H[:, k].T @ codebook_BS[:, id1] + noise
        a1 = np.abs(temp1) ** 2

        noise = sigma * (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)

        temp2 = H[:, k].T @ codebook_BS[:, id2] + noise
        a2 = np.abs(temp2) ** 2

        if a1 >= a2:
            id_BS[k] = id1
            array_gain[k] = np.abs(H[:, k].T @ codebook_BS[:, id1]) ** 2
        else:
            id_BS[k] = id2
            array_gain[k] = np.abs(H[:, k].T @ codebook_BS[:, id2]) ** 2

    return array_gain, id_BS
