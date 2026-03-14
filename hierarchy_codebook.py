import numpy as np

def hierarchy_codebook(Nt):

    l = int(np.log2(Nt) - 1)

    w_hierarchy_far = np.zeros((l, 2, Nt), dtype=complex)

    index = np.zeros((2 * l, Nt // 2), dtype=int)

    temp_codebook = np.zeros((l, Nt), dtype=int)

    # Generate binary patterns
    for i in range(1, Nt // 2 + 1):

        binary = list(map(int, format(i - 1, f'0{l}b')))

        temp_codebook[:, 2 * i - 2] = binary
        temp_codebook[:, 2 * i - 1] = binary

    # Find indices of 0s and 1s
    for i in range(l):

        index[2 * i] = np.where(temp_codebook[i, :] == 0)[0]
        index[2 * i + 1] = np.where(temp_codebook[i, :] == 1)[0]

    # Generate wide beams
    for i in range(l):

        w_hierarchy_far[i, 0, :] = generate_widebeam(Nt, index[2 * i])
        w_hierarchy_far[i, 1, :] = generate_widebeam(Nt, index[2 * i + 1])

    return w_hierarchy_far
