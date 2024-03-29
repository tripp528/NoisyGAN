import os
from absl import logging
import glob
import matplotlib.pyplot as plt
import numpy as np
import librosa, librosa.display # display explicitly, bug https://github.com/librosa/librosa/issues/343
import tensorflow as tf

import ddsp
import ddsp.training
from ddsp.training.data_preparation.prepare_tfrecord_lib import prepare_tfrecord

logging.set_verbosity(logging.INFO)
plt.style.use("dark_background")
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
    import IPython.display as ipd # import here to not install on vacc
    return ipd.display(ipd.Audio(audio, rate=sr))

def describeSample(sample):
    for key in sample.keys():
        print(key + ":", sample[key].shape, \
                # "Range:", \
                # tf.keras.backend.min(sample[key]),
                # tf.keras.backend.max(sample[key]),\
                # "type:", type(sample[key]), \
                )

def find_model_dir(dir_name):
    # Iterate through directories until model directory is found
    for root, dirs, filenames in os.walk(dir_name):
        for filename in filenames:
            # TODO: Figure out why gin is here and take it out
            if filename.endswith(".gin") and not filename.startswith("."):
                model_dir = root
                logging.info("found " + model_dir)
                return model_dir, True

        # empty model dir
        return dir_name, False

    # no model dir
    return dir_name, False

def maybe_make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def buildTFRecords(audio_input, output_tfrecord_path):
    logging.info("Building TFRecords")

    if not glob.glob(audio_input):
        raise ValueError('No audio files found. Please use the previous cell to '
                        'upload.')
    else:
        logging.info("found " + audio_input)

    input_audio_paths = []
    input_audio_paths.extend(tf.io.gfile.glob(audio_input))

    prepare_tfrecord(
        input_audio_paths,
        output_tfrecord_path,
        num_shards=10,
        pipeline_options='--runner=DirectRunner')

def merge(a, b, path=None):
    "merges b into a"
    if path is None:
        path = []
    for key in b:
        if (key in a) and (isinstance(a[key], dict) and isinstance(b[key], dict)):
            merge(a[key], b[key], path + [str(key)])
        else:
            a[key] = b[key]
    return a
