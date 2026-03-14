import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------
# PARAMETERS
# -----------------------------
Nt = 256          # Number of antennas
f = 60e9          # Frequency (Hz)
c = 3e8           # Speed of light (m/s)
K = 1             # Number of users
L = 1             # Number of paths (LOS first)
_lambda = c / f
d = _lambda / 2
N_iter = 1000

# Antenna codebooks
w_far_BS = exhaustive_codebook(Nt)
w_hierarchy_far = hierarchy_codebook(Nt)
w_hierarchy_tra = hierarchy_binary_codebook(Nt)
conv_encoder_conf = ConvEncoderConf()
Codebook = hierarchy_conv_codebook(Nt, conv_encoder_conf)

# -----------------------------
# DISTANCE LIST & SNR
# -----------------------------
dis_list = np.arange(40, 201, 20)  # Distance from 40m to 200m in steps of 20
SNR_dB_list = np.array([commParams(d) for d in dis_list])  # Custom function mapping distance to SNR_dB
SNR_list = 10 ** (SNR_dB_list / 10)
SNR_len = len(SNR_list)

# -----------------------------
# Initialize success rate arrays
# -----------------------------
r_tra0 = np.zeros(SNR_len)
r_far_exhaustive0 = np.zeros(SNR_len)
r_softconv_hierarchy0 = np.zeros(SNR_len)
r_repeat_hierarchy0 = np.zeros(SNR_len)

# -----------------------------
# SIMULATION
# -----------------------------
start_time = time.time()

for idx, SNR in enumerate(SNR_list):
    print(f"SNR_dB = {SNR_dB_list[idx]:.2f} [{idx+1}/{SNR_len}] | elapsed {time.time() - start_time:.2f} s")
    
    # Initialize success counters for each iteration
    r_tra = np.zeros(N_iter)
    r_far_exhaustive = np.zeros(N_iter)
    r_softconv_hierarchy = np.zeros(N_iter)
    r_repeat_hierarchy = np.zeros(N_iter)
    
    # Generate channels for all iterations
    H = np.zeros((Nt, K, N_iter), dtype=complex)
    h_BS = np.zeros((Nt, K, N_iter), dtype=complex)
    theta_DoA = np.zeros((N_iter, K))
    
    for i_iter in range(N_iter):
        for k in range(K):
            theta_DoA[i_iter, k] = np.random.randint(1, Nt+1)
            H[:, k, i_iter], h_BS[:, k, i_iter] = generate_channel(Nt, theta_DoA[i_iter, k], d, _lambda)
        
        # Optimal beam (for success comparison)
        array_gain_opt = np.zeros(K)
        id_BS_opt = np.zeros(K, dtype=int)
        for k in range(K):
            array_gain_opt[k], id_BS_opt[k] = training_exhaustive(w_far_BS, H[:, k, i_iter], 1e5, Nt)
        
        # Exhaustive BT
        for k in range(K):
            array_gain, id_BS = training_exhaustive(w_far_BS, H[:, k, i_iter], SNR, Nt)
            r_far_exhaustive[i_iter] = r_far_exhaustive[i_iter] or (id_BS == id_BS_opt[k])
        
        # Traditional hierarchical BT
        for k in range(K):
            array_gain, id_BS = training_hierarchy_tra(w_hierarchy_tra, H[:, k, i_iter], SNR, Nt)
            r_tra[i_iter] = r_tra[i_iter] or (id_BS == id_BS_opt[k])
        
        # Repetitive code-based BT
        array_gain, id_BS = training_hierarchy_repeat(w_hierarchy_far, Nt, H[:, :, i_iter], K, SNR)
        for k in range(K):
            r_repeat_hierarchy[i_iter] = r_repeat_hierarchy[i_iter] or (id_BS[k] == id_BS_opt[k])
        
        # Proposed CBT (adaptive softconv)
        for k in range(K):
            array_gain, id_BS = training_hierarchy_softconv(Codebook, H[:, k, i_iter], SNR, Nt, conv_encoder_conf)
            r_softconv_hierarchy[i_iter] = r_softconv_hierarchy[i_iter] or (id_BS == id_BS_opt[k])
    
    # Compute success rates
    r_tra0[idx] = np.sum(r_tra) / N_iter
    r_far_exhaustive0[idx] = np.sum(r_far_exhaustive) / N_iter
    r_softconv_hierarchy0[idx] = np.sum(r_softconv_hierarchy) / N_iter
    r_repeat_hierarchy0[idx] = np.sum(r_repeat_hierarchy) / N_iter

# -----------------------------
# PLOT SUCCESS RATE VS DISTANCE
# -----------------------------
plt.figure(figsize=(8,6))
plt.grid(True, linestyle='--', linewidth=0.5)

plt.plot(dis_list, r_softconv_hierarchy0, '-+', label='Proposed CBT', linewidth=1.6)
plt.plot(dis_list, r_tra0, '-o', label='Traditional hierarchical BT', linewidth=1.6)
plt.plot(dis_list, r_far_exhaustive0, '-s', label='Exhaustive BT', linewidth=1.6)
plt.plot(dis_list, r_repeat_hierarchy0, '-d', label='Repetitive code-based BT', linewidth=1.6)

plt.xlabel('Distance (m)', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)
plt.legend(fontsize=12)
plt.title('Beam Training Success Rate vs UE Distance')
plt.show()
