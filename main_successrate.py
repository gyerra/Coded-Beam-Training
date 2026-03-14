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

SNR_dB_list = np.arange(-10, 12, 2)
SNR_list = 10 ** (SNR_dB_list / 10)
SNR_len = len(SNR_list)

# -----------------------------
# CODEBOOKS
# -----------------------------
w_far_BS = exhaustive_codebook(Nt)
w_hierarchy_far = hierarchy_codebook(Nt)
w_hierarchy_tra = hierarchy_binary_codebook(Nt)

conv_encoder_conf = ConvEncoderConf()
Codebook = hierarchy_conv_codebook(Nt, conv_encoder_conf)

# -----------------------------
# INITIALIZE SUCCESS RATE MATRICES
# -----------------------------
r_tra = np.zeros(SNR_len)
r_far_exhaustive = np.zeros(SNR_len)
r_softconv_hierarchy = np.zeros(SNR_len)
r_repeat_hierarchy = np.zeros(SNR_len)

# -----------------------------
# SIMULATION LOOP
# -----------------------------
start_time = time.time()

for idx, SNR_dB in enumerate(SNR_dB_list):
    SNR = SNR_list[idx]
    print(f"SNR_dB = {SNR_dB} [{idx+1}/{SNR_len}] | run {time.time() - start_time:.2f} s")
    
    for i_iter in range(N_iter):
        # Random DoA for each user
        theta_DoA = np.random.randint(1, Nt+1, size=K)
        H = np.zeros((Nt, K), dtype=complex)
        h_BS = np.zeros((Nt, K), dtype=complex)

        for k in range(K):
            H[:, k], h_BS[:, k] = generate_channel(Nt, theta_DoA[k], d, _lambda)

        # -----------------------------
        # OPTIMAL BEAM (benchmark)
        # -----------------------------
        array_gain_opt = np.zeros(K)
        id_BS_opt = np.zeros(K, dtype=int)
        for k in range(K):
            array_gain_opt[k], id_BS_opt[k] = training_exhaustive(w_far_BS, H[:, k], 1e5, Nt)

        # -----------------------------
        # Exhaustive BT
        # -----------------------------
        for k in range(K):
            array_gain_exhaustive, id_BS_exhaustive = training_exhaustive(w_far_BS, H[:, k], SNR, Nt)
            if id_BS_exhaustive == id_BS_opt[k]:
                r_far_exhaustive[idx] += 1

        # -----------------------------
        # Traditional hierarchical BT
        # -----------------------------
        for k in range(K):
            array_gain_tra, id_BS_tra = training_hierarchy_tra(w_hierarchy_tra, H[:, k], SNR, Nt)
            if id_BS_tra == id_BS_opt[k]:
                r_tra[idx] += 1

        # -----------------------------
        # Repetitive code-based BT
        # -----------------------------
        array_gain_repeat, id_BS_repeat = training_hierarchy_repeat(w_hierarchy_far, Nt, H, K, SNR)
        for k in range(K):
            if id_BS_repeat[k] == id_BS_opt[k]:
                r_repeat_hierarchy[idx] += 1

        # -----------------------------
        # Proposed CBT (adaptive soft)
        # -----------------------------
        for k in range(K):
            array_gain_softconv, id_BS_softconv = training_hierarchy_softconv(Codebook, H[:, k], SNR, Nt, conv_encoder_conf)
            if id_BS_softconv == id_BS_opt[k]:
                r_softconv_hierarchy[idx] += 1

# -----------------------------
# NORMALIZE SUCCESS RATE
# -----------------------------
r_tra /= (K * N_iter)
r_far_exhaustive /= (K * N_iter)
r_softconv_hierarchy /= (K * N_iter)
r_repeat_hierarchy /= (K * N_iter)

# -----------------------------
# PLOTTING
# -----------------------------
plt.figure(figsize=(8,6))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.plot(SNR_dB_list, r_softconv_hierarchy, '-+', label='Proposed CBT', linewidth=1.6)
plt.plot(SNR_dB_list, r_tra, '-o', label='Traditional hierarchical BT', linewidth=1.6)
plt.plot(SNR_dB_list, r_far_exhaustive, '-s', label='Exhaustive BT', linewidth=1.6)
plt.plot(SNR_dB_list, r_repeat_hierarchy, '-d', label='Repetitive code-based BT', linewidth=1.6)

plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)
plt.legend(fontsize=12)
plt.title('Beam Training Success Rate vs SNR')
plt.show()
