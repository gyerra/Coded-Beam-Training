import numpy as np

def generate_widebeam2(Nt, index):

    index_length = len(index)

    K = 2 * Nt

    # Expected beam pattern amplitude
    g_Omega = np.zeros((K, 1))

    for idx in range(index_length):
        g_Omega[2 * idx] = 1
        g_Omega[2 * idx + 1] = 1

    # Scan angle range
    Omega_k = (-1 + (2 * (np.arange(1, K + 1)) - 1) / K).reshape(-1, 1)

    # Steering matrix
    A = np.zeros((Nt, K), dtype=complex)

    for k in range(K):
        A[:, k] = np.exp(1j * np.pi * Omega_k[k] * (np.arange(Nt)))

    # Random phase initialization
    theta = np.exp(1j * 2 * np.pi * np.random.rand(K, 1))

    # Construct optimization matrix
    U = np.diag(g_Omega.flatten()) @ (A.conj().T @ A) @ np.diag(g_Omega.flatten())
    U = (U + U.conj().T) / 2

    # Run MM optimization
    theta = MMAlgorithm(-U, np.zeros((K, 1)), theta, 1000, 1e-5)

    # Compute beamforming vector
    g = np.diag(g_Omega.flatten()) @ theta

    v = (1 / K) * A @ g

    v = v / np.linalg.norm(v)

    return v
