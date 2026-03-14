import numpy as np

def db2pow(db):
    return 10 ** (db / 10)

def pow2db(p):
    return 10 * np.log10(p)

def mag2db(m):
    return 20 * np.log10(m)

def commParams(d_UE):

    N_BS = 256

    # Thermal noise (-174 dBm/Hz)
    n0 = db2pow(-174 - 30)

    BW = 1e9
    N_OFDM = 256

    BW_singleCarrier = BW / N_OFDM

    noiseFactor_dB = 10

    # Transmit power
    Pt = db2pow(50 - 30)

    fc = 60e9
    c = 3e8
    lamada = c / fc

    pathLossFactor = 1

    PL = mag2db(lamada / (4 * np.pi) / (d_UE ** pathLossFactor))

    GTx_dB = 3
    GRx_dB = 0

    # channel gain
    channelGain_dB = GTx_dB + PL

    recvPower_dB = pow2db(Pt / N_OFDM) + channelGain_dB

    noisePower_dB = -110

    SNR_dB = recvPower_dB - noisePower_dB

    return SNR_dB
