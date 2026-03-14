import numpy as np
from scipy.special import iv


def training_hierarchy_softconv1(w_hierarchy_conv, h, SNR, Nt, conv_encoder_conf):

    sigma = np.sqrt(1 / SNR)

    nnt = np.arange(Nt).reshape(-1, 1)
    codeword_BS = np.linspace(-1 + 1/Nt, 1 - 1/Nt, Nt)

    codebook_BS = np.exp(1j * np.pi * nnt @ codeword_BS.reshape(1, -1)) / np.sqrt(Nt)

    # convolutional encoder configuration
    n = conv_encoder_conf["n"]
    k = conv_encoder_conf["k"]
    N = conv_encoder_conf["N"]
    A = conv_encoder_conf["A"]

    N_inner_state_bits = (N - 1) * k
    N_status = 2 ** ((N - 1) * k)
    N_choices = 2 ** k

    NextStat = np.zeros((N_status, N_choices), dtype=int)
    ConvOutput = [[None]*N_choices for _ in range(N_status)]

    # build trellis
    for ns in range(N_status):
        for nc in range(N_choices):

            merged = np.concatenate((to_binary(nc, k), to_binary(ns, N_inner_state_bits)))

            NextStat[ns, nc] = from_binary(merged[:N_inner_state_bits])

            o_block = np.zeros(n)

            for o_iter in range(n):
                o_block[o_iter] = np.mod(np.dot(merged, A[o_iter]), 2)

            ConvOutput[ns][nc] = o_block

    INF = int((np.log2(Nt) - 1) / k * n)

    losses = np.ones(N_status) * 1e9
    losses[0] = 0

    N_input_blocks = int(INF / n)

    best_path_choices = [np.zeros((N_status, 2), dtype=int) for _ in range(N_input_blocks)]

    path_cache_last = np.zeros(N_status, dtype=int)
    path_cache_new = np.zeros(N_status, dtype=int)

    w_BS = np.zeros((Nt, 4), dtype=complex)

    for k_iter in range(1, N_input_blocks + 1):

        # fetch beams directly from hierarchical convolutional codebook
        w_BS[:, 0] = w_hierarchy_conv[2*k_iter-2, 0, :]
        w_BS[:, 1] = w_hierarchy_conv[2*k_iter-2, 1, :]
        w_BS[:, 2] = w_hierarchy_conv[2*k_iter-1, 0, :]
        w_BS[:, 3] = w_hierarchy_conv[2*k_iter-1, 1, :]

        noise = sigma * (np.random.randn(4) + 1j*np.random.randn(4)) / np.sqrt(2)

        temp = h.T @ w_BS + noise
        A_meas = np.abs(temp) ** 2

        new_losses = np.ones(N_status) * np.nan

        for ns in range(N_status):
            for nc in range(N_choices):

                o_block = ConvOutput[ns][nc]

                amp = np.sqrt(2)

                a1 = np.sqrt(amp**2 * A_meas[0]) * 2 / (sigma**2)
                llr1 = np.log(iv(0, a1)) - amp**2 / (sigma**2)

                a2 = np.sqrt(amp**2 * A_meas[2]) * 2 / (sigma**2)
                llr2 = np.log(iv(0, a2)) - amp**2 / (sigma**2)

                llr1 = np.clip(llr1, -4, 4)
                llr2 = np.clip(llr2, -4, 4)

                if np.array_equal(o_block, [0, 0]):
                    llr = llr1 + llr2
                elif np.array_equal(o_block, [0, 1]):
                    llr = llr1 - llr2
                elif np.array_equal(o_block, [1, 0]):
                    llr = -llr1 + llr2
                else:
                    llr = -llr1 - llr2

                t_loss = -llr

                t_ind = NextStat[ns, nc]

                candidate = losses[ns] + t_loss

                if np.isnan(new_losses[t_ind]) or candidate < new_losses[t_ind]:

                    new_losses[t_ind] = candidate
                    best_path_choices[k_iter-1][t_ind] = [nc, ns]

        losses = new_losses
        path_cache_last = path_cache_new.copy()

        for ns in range(N_status):
            input_bit, last_sta = best_path_choices[k_iter-1][ns]
            path_cache_new[ns] = path_cache_last[last_sta]*2 + input_bit

    # traceback
    decoded_bits = np.zeros(N_input_blocks * k, dtype=int)

    p = np.argmin(losses)

    for iter in range(N_input_blocks-1, -1, -1):

        o_index = k * iter
        choice, last_state = best_path_choices[iter][p]

        decoded_bits[o_index:o_index+k] = to_binary(choice, k)

        p = last_state

    id_val = from_binary(decoded_bits)

    id1 = id_val * 2
    id2 = id_val * 2 + 1

    noise = sigma * (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)

    temp1 = h.T @ codebook_BS[:, id1] + noise
    a1 = np.abs(temp1)**2

    noise = sigma * (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)

    temp2 = h.T @ codebook_BS[:, id2] + noise
    a2 = np.abs(temp2)**2

    if a1 >= a2:
        id_BS_softconv = id1
        array_gain_softcon = np.abs(h.T @ codebook_BS[:, id1])**2
    else:
        id_BS_softconv = id2
        array_gain_softcon = np.abs(h.T @ codebook_BS[:, id2])**2

    return array_gain_softcon, id_BS_softconv
