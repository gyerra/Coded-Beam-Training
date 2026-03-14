import numpy as np

def hierarchy_conv_codebook(Nt, conv_encoder_conf):

    N = int(np.log2(Nt) - 1)

    conv_encoder_trailing = False

    Codebook = np.zeros((2 * N, Nt), dtype=int)

    for i in range(1, Nt // 2 + 1):

        # Convert (i-1) to binary with length N
        binary = list(map(int, format(i - 1, f'0{N}b')))

        # Convolutional encoding
        encoded = conv_encode(binary, conv_encoder_conf)

        # Store result in two adjacent columns
        Codebook[:, 2 * i - 2] = encoded
        Codebook[:, 2 * i - 1] = encoded

    return Codebook
