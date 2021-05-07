# -*- coding: utf-8 -*-
"""
Created on Apr 14 2021 13:08

@author: Jan.Reimes
"""

import numpy as np
import librosa
from scipy.signal import lfilter
from scipy.interpolate import interp1d

def ltasP50(freq, freq_lower=None, freq_upper=None, fmin=100, fmax=8000):
    freq = np.maximum(freq, 1.0) # avoid f=0
    if freq_lower is None:
        freq_lower = freq[0]

    if freq_upper is None:
        freq_upper = freq[-1]

    # Spectral density, ITU-T P.50, clause 4.1, equation 4-1
    Sd = -376.44 + 465.439* np.log10(freq) - 157.745 * np.log10(freq)**2 + 16.7124 * np.log10(freq)**3
    Sd -= 94.0 # it's SPL?!

    # extrapolate for f < 100 and f > 8kHz (P.50 is not defined there)
    idxMin = np.argmin(np.abs(freq-fmin))
    idxMax = np.argmin(np.abs(freq-fmax))+1
    SdI = interp1d(np.log10(freq[idxMin:idxMax]), Sd[idxMin:idxMax], fill_value="extrapolate")
    Sd_i = SdI(np.log10(freq))

    delta_f = freq_upper - freq_lower
    corr_f = 10*np.log10(delta_f)
    S = Sd_i + corr_f
    return S

def applySpecSub(signal, fs, snr, **kwargs):
    # parse arguments
    overlap = kwargs.get('overlap', 0.50)
    n_fft = kwargs.get('n_fft', 1024)
    window = kwargs.get('window', 'hann')
    pow_exp = kwargs.get('pow_exp', 2.0)
    osf = kwargs.get('osf', 0.99) # 1.0: highest musical tones, less noise; 0.0: less distortions, more noise
    tcNoise = kwargs.get('tcNoise', 0.100)
    tcSpeech = kwargs.get('tcSpeech', 0.100)
    floorSubtractFactor = kwargs.get('floorSubtractFactor', 0.0)

    # check arguments
    floorSubtractFactor = np.maximum(floorSubtractFactor, 0.0)
    overlap = np.maximum(np.minimum(overlap, 0.99), 0.0)
    osf = np.maximum(np.minimum(osf, 1.0), 0.0)

    # derive parameters from arguments
    hop_length = int(n_fft * (1-overlap))
    fsBlock = fs / ((1 - overlap) * n_fft)

    # transform input
    freq = librosa.fft_frequencies(fs, n_fft)
    stft_args = dict(n_fft=n_fft, win_length=n_fft, hop_length=hop_length, window=window, center=True)
    S = librosa.stft(signal, **stft_args)
    S_avg = np.abs(S.mean(axis=1))
    levelS = 10*np.log10(np.sum(S_avg**2) / n_fft)

    # speech-shaped noise
    n = np.random.randn(signal.shape[0]).astype(np.float32)
    N = librosa.stft(n, **stft_args)

    S_ltass = ltasP50(freq)
    S_ltass = np.power(10, S_ltass/20)

    N *= np.repeat(np.reshape(S_ltass, (S_ltass.shape[0],1)), N.shape[1], axis=1)
    levelN = 10*np.log10(np.sum(np.abs(N.mean(axis=1))**2) / n_fft)

    # scale noise
    diff = (levelS - levelN) - snr
    levelN += diff
    N *= np.power(10, diff/20)

    # combine!
    Y = S + N

    # smooth
    aS = np.exp(-1/(tcSpeech * fsBlock))
    aN = np.exp(-1/(tcNoise * fsBlock))
    absY = lfilter([1-aS], [1, -aS], np.abs(Y), axis=1)
    absN = lfilter([1-aN], [1, -aN], np.abs(N), axis=1)

    # spectral subtraction
    S_est = np.maximum(absY-osf*absN, floorSubtractFactor*absY)

    # Wiener gain
    G = np.power(S_est**pow_exp/(S_est**pow_exp + absN**pow_exp), 1/pow_exp)

    # transform back to time domain
    stft_args.pop('n_fft')
    degraded = librosa.istft(S*G, **stft_args)

    # zero padding
    degraded = np.pad(degraded, (0, signal.shape[0]-degraded.shape[0])).astype(np.float32)
    return degraded

if __name__ == "__main__":
    pass
