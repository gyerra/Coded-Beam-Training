import numpy as np
from scipy.sparse.linalg import eigs

def MMAlgorithm(U, v, phi, iter_max, accuracy):

    M = len(v)

    try:
        lambda_val = np.max(np.real(eigs(U, k=1, return_eigenvectors=False)))
    except:
        lambda_val = 1e8 * np.linalg.norm(U)

    t = 1

    f_phi_new = phi.conj().T @ U @ phi + 2 * np.real(phi.conj().T @ v)

    while t <= iter_max:

        f_phi = f_phi_new

        q = (lambda_val * np.eye(M) - U) @ phi - v

        phi = np.exp(1j * np.angle(q))

        f_phi_new = phi.conj().T @ U @ phi + 2 * np.real(phi.conj().T @ v)

        if abs(f_phi - f_phi_new) / abs(f_phi_new) <= accuracy:
            break

        t += 1

    phi = np.conj(phi)

    return phi
