import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------
# PARAMETERS
# -----------------------------
Nt = 1024          # Number of BS antennas
f = 60e9           # Frequency (Hz)
c = 3e8            # Speed of light (m/s)
K = 1              # Number of users
L = 1              # Number of channel paths (LOS)
_lambda = c / f
d = _lambda / 2
N_iter = 1000
theta_min = -np.pi / 2
theta_max = np.pi / 2

SNR_dB_list = np.arange(-5, 11, 3)
SNR_list = 10 ** (SNR_dB_list / 10)
SNR_len = len(SNR_list)

# -----------------------------
# CODEBOOKS
# -----------------------------
# Placeholders for MATLAB functions
w_far_BS = exhaustive_codebook(Nt)
w_hierarchy_far = hierarchy_codebook(Nt)
w_hierarchy_tra = hierarchy_binary_codebook(Nt)

conv_encoder_conf = ConvEncoderConf()  # adaptive encoder
Codebook = hierarchy_conv_codebook(Nt, conv_encoder_conf)      # adaptive CBT
w_hierarchy_conv = hierarchy_conv_codebook1(Nt, conv_encoder_conf)  # non-adaptive CBT

# -----------------------------
# INITIALIZE RATE MATRICES
# -----------------------------
rate_softconv_hierarchy = np.zeros((SNR_len, N_iter))
rate_softconv_hierarchy1 = np.zeros((SNR_len, N_iter))
rate_conv_hierarchy = np.zeros((SNR_len, N_iter))
rate_conv_hierarchy1 = np.zeros((SNR_len, N_iter))

# -----------------------------
# SIMULATION LOOP
# -----------------------------
start_time = time.time()

for idx, SNR_dB in enumerate(SNR_dB_list):
    SNR = SNR_list[idx]
    print(f"SNR_dB = {SNR_dB} [{idx+1}/{SNR_len}] | run {time.time() - start_time:.2f} s")
    
    for i_iter in range(N_iter):
        # Random DoA for users
        theta_DoA = np.random.randint(1, Nt+1, size=K)
        H = np.zeros((Nt, K), dtype=complex)
        h_BS = np.zeros((Nt, K), dtype=complex)

        for k in range(K):
            H[:, k], h_BS[:, k] = generate_channel(Nt, theta_DoA[k], d, _lambda)

        # -----------------------------
        # Adaptive Hard CBT
        # -----------------------------
        for k in range(K):
            array_gain_conv, id_BS_conv = training_hierarchy_conv(Codebook, H[:, k], SNR, Nt, conv_encoder_conf)
            rate_conv_hierarchy[idx, i_iter] += np.log2(1 + SNR * array_gain_conv)

        # -----------------------------
        # Non-Adaptive Hard CBT
        # -----------------------------
        for k in range(K):
            array_gain_conv1, id_BS_conv1 = training_hierarchy_conv1(w_hierarchy_conv, H[:, k], SNR, Nt, conv_encoder_conf)
            rate_conv_hierarchy1[idx, i_iter] += np.log2(1 + SNR * array_gain_conv1)

        # -----------------------------
        # Adaptive Soft CBT
        # -----------------------------
        for k in range(K):
            array_gain_softconv, id_BS_softconv = training_hierarchy_softconv(Codebook, H[:, k], SNR, Nt, conv_encoder_conf)
            rate_softconv_hierarchy[idx, i_iter] += np.log2(1 + SNR * array_gain_softconv)

        # -----------------------------
        # Non-Adaptive Soft CBT
        # -----------------------------
        for k in range(K):
            array_gain_softconv1, id_BS_softconv1 = training_hierarchy_softconv1(w_hierarchy_conv, H[:, k], SNR, Nt, conv_encoder_conf)
            rate_softconv_hierarchy1[idx, i_iter] += np.log2(1 + SNR * array_gain_softconv1)

# -----------------------------
# AVERAGE OVER ITERATIONS
# -----------------------------
rate_softconv_hierarchy = rate_softconv_hierarchy.mean(axis=1)
rate_softconv_hierarchy1 = rate_softconv_hierarchy1.mean(axis=1)
rate_conv_hierarchy = rate_conv_hierarchy.mean(axis=1)
rate_conv_hierarchy1 = rate_conv_hierarchy1.mean(axis=1)

# -----------------------------
# PLOTTING
# -----------------------------
plt.figure(figsize=(8,6))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.plot(SNR_dB_list, rate_softconv_hierarchy, '-+', label='Adaptive CBT (soft)', linewidth=1.6)
plt.plot(SNR_dB_list, rate_softconv_hierarchy1, '-*', label='Non-adaptive CBT (soft)', linewidth=1.6)
plt.plot(SNR_dB_list, rate_conv_hierarchy, '-s', label='Adaptive CBT (hard)', linewidth=1.6)
plt.plot(SNR_dB_list, rate_conv_hierarchy1, '-d', label='Non-adaptive CBT (hard)', linewidth=1.6)

plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Average Rate (bit/s/Hz)', fontsize=12)
plt.legend(fontsize=12)
plt.title('CBT Performance Comparison: Adaptive vs Non-Adaptive, Soft vs Hard Decoding')
plt.show()
