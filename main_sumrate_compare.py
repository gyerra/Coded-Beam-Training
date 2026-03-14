import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------
# PARAMETERS
# -----------------------------
Nt = 1024           # Number of BS antennas
f = 60e9            # Frequency (Hz)
c = 3e8             # Speed of light (m/s)
K = 1               # Number of users
L = 1               # Number of channel paths (LOS)
_lambda = c / f
d = _lambda / 2

N_iter = 1000
theta_min = -np.pi/2
theta_max = np.pi/2

SNR_dB_list = np.arange(-5, 11, 3)  # -5 dB to 10 dB
SNR_list = 10 ** (SNR_dB_list / 10)
SNR_len = len(SNR_list)

# -----------------------------
# CODEBOOKS
# -----------------------------
# Note: You need to implement these functions in Python
# They generate beamforming codebooks

w_far_BS = exhaustive_codebook(Nt)          # exhaustive search
w_hierarchy_far = hierarchy_codebook(Nt)    # wide/narrow/medium beams
w_hierarchy_tra = hierarchy_binary_codebook(Nt)  # binary splitting
conv_encoder_conf = ConvEncoderConf()       # your previous encoder class
Codebook = hierarchy_conv_codebook(Nt, conv_encoder_conf)  # coded beam training

# -----------------------------
# INITIALIZE DATA RATE MATRICES
# -----------------------------
rate_far_exhaustive = np.zeros((SNR_len, N_iter))
rate_tra = np.zeros((SNR_len, N_iter))
rate_softconv_hierarchy = np.zeros((SNR_len, N_iter))
rate_repeat_hierarchy = np.zeros((SNR_len, N_iter))

# -----------------------------
# SIMULATION LOOP
# -----------------------------
start_time = time.time()

for idx, SNR_dB in enumerate(SNR_dB_list):
    SNR = SNR_list[idx]
    print(f"SNR_dB = {SNR_dB} [{idx+1}/{SNR_len}] | run {time.time() - start_time:.2f} s")
    
    for i_iter in range(N_iter):
        theta_DoA = np.random.randint(1, Nt+1, size=K)  # random direction for each user
        H = np.zeros((Nt, K), dtype=complex)
        h_BS = np.zeros((Nt, K), dtype=complex)

        # Generate channel for each user
        for k in range(K):
            H[:, k], h_BS[:, k] = generate_channel(Nt, theta_DoA[k], d, _lambda)

        # -----------------------------
        # EXHAUSTIVE TRAINING
        # -----------------------------
        for k in range(K):
            array_gain_exhaustive, id_BS_exhaustive = training_exhaustive(w_far_BS, H[:, k], SNR, Nt)
            rate_far_exhaustive[idx, i_iter] += np.log2(1 + SNR * array_gain_exhaustive)

        # -----------------------------
        # TRADITIONAL HIERARCHICAL TRAINING
        # -----------------------------
        for k in range(K):
            array_gain_tra, id_BS_tra = training_hierarchy_tra(w_hierarchy_tra, H[:, k], SNR, Nt)
            rate_tra[idx, i_iter] += np.log2(1 + SNR * array_gain_tra)

        # -----------------------------
        # REPETITIVE HIERARCHICAL TRAINING
        # -----------------------------
        array_gain_repeat, id_BS_repeat = training_hierarchy_repeat(w_hierarchy_far, Nt, H, K, SNR)
        for k in range(K):
            rate_repeat_hierarchy[idx, i_iter] += np.log2(1 + SNR * array_gain_repeat[k])

        # -----------------------------
        # PROPOSED CBT (CONVOLUTIONAL CODED BEAM TRAINING)
        # -----------------------------
        for k in range(K):
            array_gain_softconv, id_BS_softconv = training_hierarchy_softconv(Codebook, H[:, k], SNR, Nt, conv_encoder_conf)
            rate_softconv_hierarchy[idx, i_iter] += np.log2(1 + SNR * array_gain_softconv)

# -----------------------------
# AVERAGE OVER ITERATIONS
# -----------------------------
rate_tra = rate_tra.mean(axis=1)
rate_far_exhaustive = rate_far_exhaustive.mean(axis=1)
rate_softconv_hierarchy = rate_softconv_hierarchy.mean(axis=1)
rate_repeat_hierarchy = rate_repeat_hierarchy.mean(axis=1)

# -----------------------------
# PLOTTING
# -----------------------------
plt.figure(figsize=(8,6))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.plot(SNR_dB_list, rate_softconv_hierarchy, '-+', label='Proposed CBT', linewidth=1.6)
plt.plot(SNR_dB_list, rate_tra, '-o', label='Traditional hierarchical BT', linewidth=1.6)
plt.plot(SNR_dB_list, rate_far_exhaustive, '-s', label='Exhaustive BT', linewidth=1.6)
plt.plot(SNR_dB_list, rate_repeat_hierarchy, '-d', label='Repetitive code-based BT', linewidth=1.6)

plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Average Rate (bit/s/Hz)', fontsize=12)
plt.legend(fontsize=12)
plt.title('Data Rate Comparison of Beam Training Algorithms')
plt.show()
