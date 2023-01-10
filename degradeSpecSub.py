# -*- coding: utf-8 -*-
"""
Created on Apr 14 2021 13:08

@author: Jan.Reimes
"""

import numpy as np
import librosa
from scipy.signal import lfilter

from helper.ltass import ltassP50FB

def applySpecSub(signal, fs, speechLevel, snr, **kwargs):
    # parse arguments
    overlap = kwargs.get('overlap', 0.75)
    n_fft = kwargs.get('n_fft', 8192)
    window = kwargs.get('window', 'hann')

    pow_exp = kwargs.get('pow_exp', 2.0)
    osf = kwargs.get('osf', 0.99) # 1.0: highest musical tones, less noise; 0.0: less distortions, more noise
    tcNoise = kwargs.get('tcNoise', 0.100)
    tcSpeech = kwargs.get('tcSpeech', 0.100)
    floorSubtractFactor = kwargs.get('floorSubtractFactor', 0.0)

    # check arguments
    floorSubtractFactor = np.maximum(floorSubtractFactor, 0.0)
    overlap = np.maximum(np.minimum(overlap, 0.99), 0.0)
    osf = np.maximum(np.minimum(osf, 2.0), 0.0)

    # derive parameters from arguments
    hop_length = int(n_fft * (1-overlap))
    fsBlock = fs / ((1 - overlap) * n_fft)

    # transform input
    freq = librosa.fft_frequencies(sr=fs, n_fft=n_fft)
    stft_args = dict(n_fft=n_fft, win_length=n_fft, hop_length=hop_length, window=window, center=True)
    S = librosa.stft(signal, **stft_args)

    # generate white noise at 0 dB
    n = np.random.randn(signal.shape[0]).astype(np.float32)
    N = librosa.stft(n, **stft_args)

    # generate speech-shaped noise at target level
    targetNoiseLevel = speechLevel - snr
    S_ltass = ltassP50FB(freq, targetLevelDbPa=targetNoiseLevel)

    S_ltass = np.power(10, S_ltass/20)
    N *= np.repeat(np.reshape(S_ltass, (S_ltass.shape[0],1)), N.shape[1], axis=1)

    # combine!
    Y = S + N

    # smooth
    aS = np.exp(-1/(tcSpeech * fsBlock))
    aN = np.exp(-1/(tcNoise * fsBlock))
    absY = lfilter([1-aS], [1, -aS], np.abs(Y), axis=1)
    absN = lfilter([1-aN], [1, -aN], np.abs(N), axis=1)

    # spectral subtraction, taking into account over-subtraction and minimum noise floor
    S_est = np.maximum(absY-osf*absN, floorSubtractFactor*absY)

    # Wiener gain
    G = np.power(S_est**pow_exp/(S_est**pow_exp + absN**pow_exp), 1/pow_exp)

    # Processed signal/STFT
    P = S * G

    # transform back to time domain
    stft_args.pop('n_fft')
    degraded = librosa.istft(P, **stft_args)

    # zero padding
    degraded = np.pad(degraded, (0, signal.shape[0]-degraded.shape[0])).astype(np.float32)

    return degraded

if __name__ == "__main__":
    pass
