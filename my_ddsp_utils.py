import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
DEFAULT_SAMPLE_RATE = 16000


def plotControls(amplitudes, harmonic_distribution, f0_hz):
    '''Plots the controls (inputs) to a ddsp processor'''

    time = np.linspace(0, n_samples / sample_rate, n_frames)

    plt.figure(figsize=(18, 4))
    plt.subplot(131)
    plt.plot(time, amps[0, :, 0])
    plt.xticks([0, 1, 2, 3, 4])
    plt.title('Amplitude')

    plt.subplot(132)
    plt.plot(time, harmonic_distribution[0, :, :])
    plt.xticks([0, 1, 2, 3, 4])
    plt.title('Harmonic Distribution')

    plt.subplot(133)
    plt.plot(time, f0_hz[0, :, 0])
    plt.xticks([0, 1, 2, 3, 4])
    _ = plt.title('Fundamental Frequency')

def specPlot(audio, sr=DEFAULT_SAMPLE_RATE):
    '''takes a tensor as input (from ddsp)'''
    # short term forier transform
    X = librosa.stft(audio.numpy().squeeze())
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.show()

def wavePlot(audio):
    '''takes a tensor as input (from ddsp)'''
    # plot waveform
    librosa.display.waveplot(audio.numpy().squeeze())
    plt.show()

def play(audio, sr=DEFAULT_SAMPLE_RATE):
    return ipd.Audio(audio, rate=sample_rate)
