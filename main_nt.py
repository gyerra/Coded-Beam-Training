import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------
# PARAMETERS
# -----------------------------
f = 60e9           # Frequency (Hz)
c = 3e8            # Speed of light (m/s)
K = 1              # Number of users
L = 1              # Number of channel paths (LOS)
_lambda = c / f
d = _lambda / 2
N_iter = 1000
theta_min = -np.pi / 2
theta_max = np.pi / 2
SNR_dB = 5
SNR = 10 ** (SNR_dB / 10)

Nt_list = [64, 128, 256, 512, 1024, 2048]
Nt_len = len(Nt_list)

# -----------------------------
# INITIALIZE DATA RATE MATRICES
# -----------------------------
rate_far_exhaustive = np.zeros((Nt_len, N_iter))
rate_tra = np.zeros((Nt_len, N_iter))
rate_softconv_hierarchy = np.zeros((Nt_len, N_iter))
rate_repeat_hierarchy = np.zeros((Nt_len, N_iter))

# -----------------------------
# SIMULATION
# -----------------------------
start_time = time.time()

for idx, Nt in enumerate(Nt_list):
    print(f"Simulating Nt = {Nt} [{idx+1}/{Nt_len}] | elapsed {time.time()-start_time:.2f} s")
    
    # Generate codebooks for this Nt
    w_far_BS = exhaustive_codebook(Nt)
    w_hierarchy_far = hierarchy_codebook(Nt)
    w_hierarchy_tra = hierarchy_binary_codebook(Nt)
    
    conv_encoder_conf = ConvEncoderConf()
    Codebook = hierarchy_conv_codebook(Nt, conv_encoder_conf)
    
    for i_iter in range(N_iter):
        H = np.zeros((Nt, K), dtype=complex)
        h_BS = np.zeros((Nt, K), dtype=complex)
        theta_DoA = np.random.randint(1, Nt+1, size=K)
        
        for k in range(K):
            H[:, k], h_BS[:, k] = generate_channel(Nt, theta_DoA[k], d, _lambda)
        
        # Exhaustive BT
        for k in range(K):
            array_gain_exhaustive, id_BS_exhaustive = training_exhaustive(w_far_BS, H[:, k], SNR, Nt)
            rate_far_exhaustive[idx, i_iter] += np.log2(1 + SNR * array_gain_exhaustive)
        
        # Traditional hierarchical BT
        for k in range(K):
            array_gain_tra, id_BS_tra = training_hierarchy_tra(w_hierarchy_tra, H[:, k], SNR, Nt)
            rate_tra[idx, i_iter] += np.log2(1 + SNR * array_gain_tra)
        
        # Repetitive code-based BT (hard decoding)
        array_gain_repeat, id_BS_repeat = training_hierarchy_repeat(w_hierarchy_far, Nt, H, K, SNR)
        for k in range(K):
            rate_repeat_hierarchy[idx, i_iter] += np.log2(1 + SNR * array_gain_repeat[k])
        
        # Proposed CBT (adaptive soft decoding)
        for k in range(K):
            array_gain_softconv, id_BS_softconv = training_hierarchy_softconv(Codebook, H[:, k], SNR, Nt, conv_encoder_conf)
            rate_softconv_hierarchy[idx, i_iter] += np.log2(1 + SNR * array_gain_softconv)

# -----------------------------
# AVERAGE OVER ITERATIONS
# -----------------------------
rate_far_exhaustive = np.mean(rate_far_exhaustive, axis=1)
rate_tra = np.mean(rate_tra, axis=1)
rate_softconv_hierarchy = np.mean(rate_softconv_hierarchy, axis=1)
rate_repeat_hierarchy = np.mean(rate_repeat_hierarchy, axis=1)

# -----------------------------
# PLOT RESULTS
# -----------------------------
plt.figure(figsize=(8,6))
plt.grid(True, linestyle='--', linewidth=0.5)

log2_Nt = np.log2(Nt_list)

plt.plot(log2_Nt, rate_softconv_hierarchy, '-+', label='Proposed CBT', linewidth=1.6)
plt.plot(log2_Nt, rate_tra, '-o', label='Traditional hierarchical BT', linewidth=1.6)
plt.plot(log2_Nt, rate_far_exhaustive, '-s', label='Exhaustive BT', linewidth=1.6)
plt.plot(log2_Nt, rate_repeat_hierarchy, '-d', label='Repetitive code-based BT', linewidth=1.6)

plt.xlabel('log2(Nt)', fontsize=12)
plt.ylabel('Average Rate (bit/s/Hz)', fontsize=12)
plt.legend(fontsize=12)
plt.title('Achievable Data Rate vs Number of Antennas')
plt.show()
