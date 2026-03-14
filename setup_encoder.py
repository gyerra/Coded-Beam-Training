import numpy as np

# Configuration for convolutional encoder
class ConvEncoderConf:
    def __init__(self):
        self.n = 2  # Output bits
        self.k = 1  # Input bits
        # For every input bit, produce 2 output bits
        # input : 1, output : 11
        # Extra bits help detect and correct errors
        # Code rate = input bits / output bits = 1/2

        self.N = 3  # Constraint length (memory: current bit + previous bits)
        self.window_factor = 6  # Useful for Viterbi decoder
        self.trailing = False  # Do not add extra bits at the end

        # Define generator polynomials for output bits
        # Each row defines XOR taps for that output bit
        self.A = [
            [1, 0, 1],  # Output bit 1
            [1, 1, 1]   # Output bit 2
        ]

        # Loss function when comparing sequences
        self.loss_func = self.hamming_distance

    # Hamming distance function
    @staticmethod
    def hamming_distance(x, y):
        assert len(x) == len(y), "Sequences must have same length"
        # XOR the two sequences and sum to get distance
        return np.sum(np.bitwise_xor(x, y))


# Example usage:
conv_encoder_conf = ConvEncoderConf()

x = np.array([1, 1, 0, 1])
y = np.array([1, 0, 0, 1])
distance = conv_encoder_conf.hamming_distance(x, y)
print("Hamming distance:", distance)
