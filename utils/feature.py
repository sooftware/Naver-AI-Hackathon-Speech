"""
  -*- coding: utf-8 -*-

  * Feature Extraction for Speech Recognition

  get_librosa_melspectrogram : get Mel-Spectrogram feature using librosa library
  get_librosa_mfcc : get MFCC (Mel-Frequency-Cepstral-Coefficient) feature using librosa library

  FRAME_LENGTH : 21ms
  STRIDE : 5.2ms ( 75% duplicated )
  FRAME_LENGTH = N_FFT / SAMPLE_RATE => N_FFT = 336
  STRIDE = HOP_LENGTH / SAMPLE_RATE => STRIDE = 168

"""

import torch
import librosa

SAMPLE_RATE = 16000
N_FFT = 336
HOP_LENGTH = 84
N_MELS = 80

def get_librosa_melspectrogram(filepath, n_mels = 80, log_ = False):
    sig, sr = librosa.core.load(filepath, SAMPLE_RATE)
    mel_spectrogram = librosa.feature.melspectrogram(sig, n_mels = N_MELS, n_fft = N_FFT, hop_length = HOP_LENGTH)

    if log_:
        log_mel = librosa.amplitude_to_db(mel_spectrogram, ref = np.max)
        log_mel = torch.FloatTensor(log_mel).transpose(0, 1)
        return log_mel
    else:
        mel_spectrogram = torch.FloatTensor(mel_spectrogram).transpose(0, 1)
        return mel_spectrogram


def get_librosa_mfcc(filepath, n_mfcc = 40):
    sig, sr = librosa.core.load(filepath, SAMPLE_RATE)
    mfccs = librosa.feature.mfcc(y = sig, sr = sr, hop_length = HOP_LENGTH, n_mfcc = n_mfcc, n_fft = N_FFT)
    mfccs = torch.FloatTensor(mfccs).transpose(0, 1)
    return mfccs
