import os

from absl import logging
# see all logging messages
logging.set_verbosity(logging.INFO)

import matplotlib.pyplot as plt
plt.style.use("dark_background")
import numpy as np
# import warnings
# warnings.filterwarnings("ignore")

import librosa, librosa.display # display explicitly, bug https://github.com/librosa/librosa/issues/343

DEFAULT_SAMPLE_RATE = 16000 # how many samples per second
DEFAULT_N_SAMPLES = DEFAULT_SAMPLE_RATE * 4 # each sample is 4 seconds by default.


def plotControls(amplitudes, harmonic_distribution, f0_hz):
    '''Plots the controls (inputs) to a ddsp processor'''

    time = np.linspace(0, n_samples / DEFAULT_SAMPLE_RATE, n_frames)

    plt.figure(figsize=(18, 4))
    plt.subplot(131)
    plt.plot(time, amplitudes[0, :, 0])
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
    '''takes a tensor as input (from ddsp)'''
    import IPython.display as ipd
    return ipd.Audio(audio, rate=sr)

def find_model_dir(dir_name):
    # Iterate through directories until model directory is found
    for root, dirs, filenames in os.walk(dir_name):
        for filename in filenames:
            # TODO: Figure out why gin is here and take it out 
            if filename.endswith(".gin") and not filename.startswith("."):
                model_dir = root
                logging.info("found " + model_dir)
                break
        return model_dir
