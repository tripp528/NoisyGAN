import glob
import time
import os
import subprocess

import tensorflow.compat.v1 as tf
logging.info("tf versionn: " + str(tf.__version__))
# tf.compat.v1.enable_v2_behavior()
import tensorflow_datasets as tfds

import ddsp
import ddsp.training
from ddsp.training.data_preparation.prepare_tfrecord_lib import prepare_tfrecord

# local imports
from my_ddsp_utils import *

class DDSP_DATASET:
    def __init__(self,output_tfrecord_path, audio_input=None):
        """
        output_tfrecord_path: filepattern string (not list)
        audio_input: filepattern string (not list)

        TODO: Make these both lists of filepatterns
        """
        self.audio_input = audio_input
        self.output_tfrecord_path = output_tfrecord_path
        self.train_file_pattern = self.output_tfrecord_path+"*"

        # build the TFRecords
        if audio_input:
            self.buildTFRecords()

        # get the data provider
        self.data_provider = ddsp.training.data.TFRecordProvider(self.train_file_pattern)
        self.n_samples = self.get_n_samples()

    def buildTFRecords(self):
        logging.info("Building TFRecords")

        if not glob.glob(self.audio_input):
            raise ValueError('No audio files found. Please use the previous cell to '
                            'upload.')
        else:
            logging.info("found " + self.audio_input)

        input_audio_paths = []
        input_audio_paths.extend(tf.io.gfile.glob(self.audio_input))

        prepare_tfrecord(
            input_audio_paths,
            self.output_tfrecord_path,
            num_shards=10,
            pipeline_options='--runner=DirectRunner')

    def getDataset(self):
        return self.data_provider.get_batch(batch_size=1, shuffle=False).take(1).repeat()

    def getSample(self,sampleNum=0):
        samples = list(iter(self.data_provider.get_dataset(shuffle=False)))
        sample = samples[sampleNum]
        return sample

    def getAudio(self,sampleNum=0):
        samples = list(iter(self.data_provider.get_dataset(shuffle=False)))
        sample = samples[sampleNum]
        audio = sample["audio"]
        return audio

    def get_n_samples(self):
        return self.getAudio().shape[0]
