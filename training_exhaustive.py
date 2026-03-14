import numpy as np

def training_exhaustive(w_far_BS, H, SNR, Nt):

    sigma = np.sqrt(1 / SNR)

    noise = sigma * (np.random.randn(Nt) + 1j*np.random.randn(Nt)) / np.sqrt(2)

    temp = H.T @ w_far_BS + noise

    A = np.abs(temp) ** 2

    array_gain = np.max(A)

    idy = np.argmax(A)

    array_gain = np.abs(H.T @ w_far_BS[:, idy]) ** 2

    return array_gain, idy
