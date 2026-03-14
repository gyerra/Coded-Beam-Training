import numpy as np


def hierarchy_conv_codebook1(Nt, conv_encoder_conf):

    N = int(np.log2(Nt) - 1)

    conv_encoder_trailing = False

    w_hierarchy_conv = np.zeros((2 * N, 2, Nt), dtype=complex)

    index = np.zeros((4 * N, Nt // 2), dtype=int)

    Codebook = np.zeros((2 * N, Nt), dtype=int)

    # Generate convolutional codewords
    for i in range(1, Nt // 2 + 1):

        # binary representation
        binary = list(map(int, format(i - 1, f'0{N}b')))

        encoded = conv_encode(binary, conv_encoder_conf)

        Codebook[:, 2 * i - 2] = encoded
        Codebook[:, 2 * i - 1] = encoded

    # Find indices of 0 and 1
    for i in range(2 * N):

        index[2 * i] = np.where(Codebook[i, :] == 0)[0]
        index[2 * i + 1] = np.where(Codebook[i, :] == 1)[0]

    # Generate wide beams
    for i in range(2 * N):

        w_hierarchy_conv[i, 0, :] = generate_widebeam(Nt, index[2 * i])
        w_hierarchy_conv[i, 1, :] = generate_widebeam(Nt, index[2 * i + 1])

    return w_hierarchy_conv
