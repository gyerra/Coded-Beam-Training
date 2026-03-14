import numpy as np

def training_hierarchy_conv1(w_hierarchy_conv, h, SNR, Nt, conv_encoder_conf):

    sigma = np.sqrt(1/SNR)

    nnt = np.arange(Nt).reshape(-1,1)
    codeword_BS = np.arange(-1+1/Nt, 1, 2/Nt)

    codebook_BS = np.exp(1j*np.pi*nnt @ codeword_BS.reshape(1,-1)) / np.sqrt(Nt)

    # Viterbi parameters
    n = conv_encoder_conf["n"]
    k = conv_encoder_conf["k"]
    N = conv_encoder_conf["N"]
    A = conv_encoder_conf["A"]

    N_inner_state_bits = (N-1)*k
    N_status = 2**((N-1)*k)
    N_choices = 2**k

    NextStat = np.zeros((N_status, N_choices), dtype=int)
    ConvOutput = [[None for _ in range(N_choices)] for _ in range(N_status)]

    # Build trellis
    for ns in range(N_status):
        for nc in range(N_choices):

            merged = np.concatenate([
                to_binary(nc, k),
                to_binary(ns, N_inner_state_bits)
            ])

            NextStat[ns, nc] = from_binary(merged[:N_inner_state_bits])

            o_block = np.zeros(n, dtype=bool)

            for o_iter in range(n):
                o_block[o_iter] = (np.dot(merged, A[o_iter]) % 2).astype(bool)

            ConvOutput[ns][nc] = o_block

    INF = int((np.log2(Nt)-1)/k*n)

    losses = INF * np.concatenate(([0], np.ones(N_status-1)))

    N_input_blocks = int(INF/n)

    best_path_choices = [
        np.zeros((N_status,2), dtype=int) for _ in range(N_input_blocks)
    ]

    path_cache_last = np.zeros(N_status)
    path_cache_new = np.zeros(N_status)

    w_BS = np.zeros((Nt,4), dtype=complex)

    for k_iter in range(1, N_input_blocks+1):

        a1 = w_hierarchy_conv[2*k_iter-2,0,:]
        w_BS[:,0] = a1

        a2 = w_hierarchy_conv[2*k_iter-2,1,:]
        w_BS[:,1] = a2

        a3 = w_hierarchy_conv[2*k_iter-1,0,:]
        w_BS[:,2] = a3

        a4 = w_hierarchy_conv[2*k_iter-1,1,:]
        w_BS[:,3] = a4

        noise = sigma*(np.random.randn(4)+1j*np.random.randn(4))/np.sqrt(2)

        temp = np.sqrt(1/2)*np.dot(h.T, w_BS) + noise
        A_val = np.abs(temp)**2

        if A_val[0] > A_val[1] and A_val[2] > A_val[3]:
            received_block = np.array([0,0])
        elif A_val[0] > A_val[1] and A_val[2] < A_val[3]:
            received_block = np.array([0,1])
        elif A_val[0] < A_val[1] and A_val[2] > A_val[3]:
            received_block = np.array([1,0])
        else:
            received_block = np.array([1,1])

        new_losses = np.full(N_status, np.nan)

        for ns in range(N_status):
            for nc in range(N_choices):

                o_block = ConvOutput[ns][nc]

                t_loss = np.sum(np.logical_xor(o_block, received_block))

                t_ind = NextStat[ns, nc]

                candidate = losses[ns] + t_loss

                if np.isnan(new_losses[t_ind]) or new_losses[t_ind] > candidate:

                    new_losses[t_ind] = candidate
                    best_path_choices[k_iter-1][t_ind,0] = nc
                    best_path_choices[k_iter-1][t_ind,1] = ns

        losses = new_losses
        path_cache_last = path_cache_new.copy()

        for ns in range(N_status):

            input_bit = best_path_choices[k_iter-1][ns,0]
            last_sta = best_path_choices[k_iter-1][ns,1]

            path_cache_new[ns] = path_cache_last[last_sta]*2 + input_bit

    decoded_bits = np.zeros(N_input_blocks*k, dtype=bool)

    if conv_encoder_conf["trailing"]:
        p = 1
    else:
        p = np.argmin(losses)

    for iter in range(N_input_blocks,0,-1):

        o_index = k*(iter-1)

        status = best_path_choices[iter-1][p]

        decoded_bits[o_index:o_index+k] = np.flip(to_binary(status[0],k))

        p = status[1]

    if conv_encoder_conf["trailing"]:
        decoded_bits = decoded_bits[:N_input_blocks*k-N_inner_state_bits]

    id_val = from_binary(decoded_bits)

    id1 = id_val*2 + 1
    id2 = id_val*2 + 2

    noise = sigma*(np.random.randn()+1j*np.random.randn())/np.sqrt(2)
    temp1 = np.dot(h.T, codebook_BS[:,id1]) + noise
    a1 = np.abs(temp1)**2

    noise = sigma*(np.random.randn()+1j*np.random.randn())/np.sqrt(2)
    temp2 = np.dot(h.T, codebook_BS[:,id2]) + noise
    a2 = np.abs(temp2)**2

    if a1 >= a2:
        id_BS_conv = id1
        array_gain_con = np.abs(np.dot(h.T, codebook_BS[:,id1]))**2
    else:
        id_BS_conv = id2
        array_gain_con = np.abs(np.dot(h.T, codebook_BS[:,id2]))**2

    return array_gain_con, id_BS_conv
