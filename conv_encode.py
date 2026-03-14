import numpy as np

def combine(i_block, registers):
    L = len(registers)
    Li = len(i_block)

    Lret = (L + 1) * Li
    ret = np.zeros(Lret, dtype=bool)

    ret[0:Li] = i_block

    for k in range(L):
        index = Li * (k + 1)
        ret[index:index + Li] = registers[k]

    return ret


def conv_encode(input_bits, conv_encoder_conf):

    n = conv_encoder_conf["n"]
    k = conv_encoder_conf["k"]
    N = conv_encoder_conf["N"]
    A = conv_encoder_conf["A"]
    trailing = conv_encoder_conf["trailing"]

    N_inner_state_bits = (N - 1) * k

    input_bits = np.array(input_bits, dtype=bool)

    # Align input bits
    n_input_bits = len(input_bits)
    temp = n_input_bits % k

    if temp != 0:
        pad = k - temp
        input_bits = np.concatenate([input_bits, np.zeros(pad, dtype=bool)])
        n_input_bits += pad

    # Calculate output size
    n_input_blocks = n_input_bits // k

    if trailing:
        n_output_blocks = n_input_blocks + N - 1
    else:
        n_output_blocks = n_input_blocks

    n_output_bits = n_output_blocks * n
    encoded_bits = np.zeros(n_output_bits, dtype=bool)

    # Initialize registers
    registers = [np.zeros(k, dtype=bool) for _ in range(N - 1)]

    # Zero padding if trailing
    if trailing:
        n_input_blocks = n_output_blocks
        input_bits = np.concatenate(
            [input_bits, np.zeros(N_inner_state_bits, dtype=bool)]
        )

    # Convolution process
    for iter in range(n_input_blocks):

        i_index = k * iter
        i_block = np.flip(input_bits[i_index:i_index + k])

        i_vct = combine(i_block, registers)

        o_block = np.zeros(n, dtype=bool)

        for o_iter in range(n):
            o_block[o_iter] = bool(
                np.mod(np.dot(i_vct, A[o_iter].T), 2)
            )

        o_index = n * iter
        encoded_bits[o_index:o_index + n] = o_block

        # Shift registers
        for r_iter in range(len(registers) - 1, 0, -1):
            registers[r_iter] = registers[r_iter - 1]

        registers[0] = i_block

    return encoded_bits
