import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------
# PARAMETERS
# -----------------------------
Nt = 1024          # Number of antennas
f = 60e9           # Frequency (Hz)
c = 3e8            # Speed of light (m/s)
K = 1              # Number of users
L = 1              # Number of paths (LOS first)
_lambda = c / f
d = _lambda / 2
N_iter = 1000

# -----------------------------
# CODEBOOKS
# -----------------------------
w_far_BS = exhaustive_codebook(Nt)          # Exhaustive search codebook
w_hierarchy_tra = hierarchy_binary_codebook(Nt)  # Traditional hierarchical codebook
conv_encoder_conf = ConvEncoderConf()       # Convolutional encoder config
Codebook = hierarchy_conv_codebook(Nt, conv_encoder_conf)  # Adaptive CBT codebook

# -----------------------------
# SNR VALUES
# -----------------------------
SNR_dB_list = np.arange(-10, 12, 2)  # -10 to 10 dB in steps of 2
SNR_list = 10 ** (SNR_dB_list / 10)
SNR_len = len(SNR_dB_list)

# -----------------------------
# SUCCESS RATE ARRAYS
# -----------------------------
r_softconv_hierarchy = np.zeros(SNR_len)
r_gau = np.zeros(SNR_len)

# -----------------------------
# SIMULATION
# -----------------------------
start_time = time.time()

for idx, SNR in enumerate(SNR_list):
    print(f"SNR_dB = {SNR_dB_list[idx]} [{idx+1}/{SNR_len}] | elapsed {time.time() - start_time:.2f} s")
    
    for i_iter in range(N_iter):
        # Generate channel
        theta_DoA = np.random.randint(1, Nt+1, size=(1, K))
        H = np.zeros((Nt, K), dtype=complex)
        h_BS = np.zeros((Nt, K), dtype=complex)
        for k in range(K):
            H[:, k], h_BS[:, k] = generate_channel(Nt, theta_DoA[0, k], d, _lambda)
        
        # Optimal beam for success comparison (very high SNR)
        array_gain_opt = np.zeros(K)
        id_BS_opt = np.zeros(K, dtype=int)
        for k in range(K):
            array_gain_opt[k], id_BS_opt[k] = training_exhaustive(w_far_BS, H[:, k], 1e5, Nt)
        
        # Proposed CBT decoder (softconv)
        for k in range(K):
            array_gain, id_BS = training_hierarchy_softconv(Codebook, H[:, k], SNR, Nt, conv_encoder_conf)
            if id_BS == id_BS_opt[k]:
                r_softconv_hierarchy[idx] += 1
        
        # Gaussian decoder
        array_gain, id_BS = training_hierarchy_gau(Codebook, H[:, k], SNR, Nt, conv_encoder_conf)
        for k in range(K):
            if id_BS[k] == id_BS_opt[k]:
                r_gau[idx] += 1

# Normalize success rate
r_softconv_hierarchy /= (K * N_iter)
r_gau /= (K * N_iter)

# -----------------------------
# PLOT SUCCESS RATE vs SNR
# -----------------------------
plt.figure(figsize=(8,6))
plt.grid(True, linestyle='--', linewidth=0.5)

plt.plot(SNR_dB_list, r_softconv_hierarchy, '-d', label='Proposed decoder', linewidth=1.6)
plt.plot(SNR_dB_list, r_gau, '-o', label='Traditional Gaussian decoder', linewidth=1.6)

plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)
plt.legend(fontsize=12)
plt.title('CBT Decoder Comparison: Success Rate vs SNR')
plt.show()
