import numpy as np

def conv_decode(encoded_bits, conv_encoder_conf, soft_decode=False):

    n = conv_encoder_conf["n"]
    k = conv_encoder_conf["k"]
    N = conv_encoder_conf["N"]
    A = conv_encoder_conf["A"]

    N_inner_state_bits = (N - 1) * k

    N_status = 2 ** ((N - 1) * k)
    N_choices = 2 ** k

    NextStat = np.zeros((N_status, N_choices), dtype=int)
    ConvOutput = [[None for _ in range(N_choices)] for _ in range(N_status)]

    # Build state transition table
    for ns in range(N_status):
        for nc in range(N_choices):

            merged = np.concatenate([
                to_binary(nc, k),
                to_binary(ns, N_inner_state_bits)
            ])

            NextStat[ns, nc] = from_binary(merged[:N_inner_state_bits])

            o_block = np.zeros(n, dtype=bool)

            for o_iter in range(n):
                o_block[o_iter] = bool(np.mod(np.dot(merged, A[o_iter].T), 2))

            ConvOutput[ns][nc] = o_block

    INF = len(encoded_bits)
    losses = INF * np.concatenate(([0], np.ones(N_status - 1)))

    if soft_decode:
        n = 1

    N_input_blocks = INF // n

    best_path_choices = [
        np.zeros((N_status, 2), dtype=int)
        for _ in range(N_input_blocks)
    ]

    for k_iter in range(N_input_blocks):

        index = n * k_iter
        received_block = encoded_bits[index:index+n]

        new_losses = np.full(N_status, np.nan)

        for ns in range(N_status):

            for nc in range(N_choices):

                o_block = ConvOutput[ns][nc]

                if soft_decode:
                    t_loss = received_block[from_binary(o_block)]
                else:
                    t_loss = np.sum(np.logical_xor(o_block, received_block))

                t_ind = NextStat[ns, nc]

                candidate_loss = losses[ns] + t_loss

                if np.isnan(new_losses[t_ind]) or new_losses[t_ind] > candidate_loss:
                    new_losses[t_ind] = candidate_loss
                    best_path_choices[k_iter][t_ind, 0] = nc
                    best_path_choices[k_iter][t_ind, 1] = ns

        losses = new_losses

    decoded_bits = np.zeros(N_input_blocks * k, dtype=bool)

    if conv_encoder_conf["trailing"]:
        p = 0
    else:
        p = np.argmin(losses)

    for iter in range(N_input_blocks - 1, -1, -1):

        o_index = k * iter
        status = best_path_choices[iter][p]

        decoded_bits[o_index:o_index+k] = np.flip(to_binary(status[0], k))

        p = status[1]

    if conv_encoder_conf["trailing"]:
        decoded_bits = decoded_bits[:N_input_blocks*k - N_inner_state_bits]

    return decoded_bits
