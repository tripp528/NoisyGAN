# local imports
from .my_ddsp_utils import *

import glob
import time
import subprocess

import tensorflow.compat.v2 as tf
logging.info("tf versionn: " + str(tf.__version__))
# tf.compat.v1.enable_v2_behavior()
import tensorflow_datasets as tfds

import ddsp
import ddsp.training
from ddsp.training.data_preparation.prepare_tfrecord_lib import prepare_tfrecord

class DDSP_DATASET:
    def __init__(self,output_tfrecord_path, audio_input=None, generator=None):
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
        self.data_provider = DDSP_TFPROV(self.train_file_pattern,generator=generator)
        # self.data_provider = ddsp.training.data.TFRecordProvider(self.train_file_pattern)
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
        howmany = len(samples)
        sample = samples[sampleNum % howmany]
        # for key in sample.keys():
        #     sample[key] = np.expand_dims(sample[key],axis=[0])
        return sample

    def getBatch(self,howmany,startNum=0):
        samples = list(iter(self.data_provider.get_dataset(shuffle=False)))
        sample = np.array(samples[startNum:howmany+startNum])
        return sample

    def getAudio(self,sampleNum=0):
        samples = list(iter(self.data_provider.get_dataset(shuffle=False)))
        sample = samples[sampleNum]
        audio = sample["audio"]
        return audio

    def get_n_samples(self):
        return self.getAudio().shape[0]


class DDSP_TFPROV(ddsp.training.data.TFRecordProvider):
    # TODO: right now it's always making thise labeled real
    def __init__(self,filepattern,label=1,generator=None):
        super().__init__(filepattern)
        self.label = label
        self.generator = generator

    # def get_batch(self, batch_size, shuffle=True, repeats=-1):
    #     """Read dataset.
    #
    #     Args:
    #     batch_size: Size of batch.
    #     shuffle: Whether to shuffle the examples.
    #     repeats: Number of times to repeat dataset. -1 for endless repeats.
    #
    #     Returns:
    #     A batched tf.data.Dataset.
    #     """
    #
    #     # true_dataset = self.get_true_dataset(shuffle)
    #     # true_dataset = true_dataset.repeat(repeats)
    #     # true_dataset = true_dataset.batch(int(batch_size/2), drop_remainder=True)
    #     # # print(type(true_dataset))
    #     # true_dataset = true_dataset.prefetch(buffer_size=ddsp.training.data._AUTOTUNE)
    #     # return true_dataset
    #
    #     fake_dataset = self.get_fake_dataset(int(batch_size/2))
    #     fake_dataset = fake_dataset.repeat(repeats)
    #     fake_dataset = fake_dataset.batch(int(batch_size/2), drop_remainder=True)
    #     fake_dataset = fake_dataset.prefetch(buffer_size=ddsp.training.data._AUTOTUNE)
    #     return fake_dataset

    def get_dataset(self, shuffle=True):
        true_dataset = self.get_true_dataset(shuffle)
        num_true = len(list(iter(true_dataset)))
        fake_dataset = self.get_fake_dataset(num_true)
        dataset = true_dataset.concatenate(fake_dataset)
        return dataset


    def get_true_dataset(self, shuffle=True):
        """Read dataset.

        Args:
        shuffle: Whether to shuffle the files.

        Returns:
        dataset: A tf.dataset that reads from the TFRecord.
        """
        def parse_tfexample(record):
            example = tf.io.parse_single_example(record, self.features_dict)
            del example["f0_confidence"]
            # for key in example.keys():
            #     example[key] = tf.expand_dims(example[key],axis=[0])

            example["label"] = tf.convert_to_tensor([1.0])
            return example

        filenames = tf.data.Dataset.list_files(self._file_pattern, shuffle=shuffle)
        dataset = filenames.interleave(
            map_func=tf.data.TFRecordDataset,
            cycle_length=40,
            num_parallel_calls=ddsp.training.data._AUTOTUNE)

        dataset = dataset.map(parse_tfexample, num_parallel_calls=ddsp.training.data._AUTOTUNE)
        return dataset

    def get_fake_dataset(self, size):
        """Read dataset.

        Args:
        shuffle: Whether to shuffle the files.

        Returns:
        dataset: A tf.dataset that reads from the TFRecord.
        """
        shapes = {"f0_hz": (1000,),
                    "loudness_db": (1000,),
                    "audio": (64000,),
                    "label": (1,)}

        def fake_sample_generator():
            for i in range(size):
                logging.info("generating... "+str(i))
                generated = self.generator.generate(label=0)
                sample = {}
                for key in shapes:
                    sample[key] = generated[key]

                yield sample

        dataset = tf.data.Dataset.from_generator(
                        fake_sample_generator,
                        output_types={k: tf.float32 for k in shapes},
                        output_shapes=shapes)

        return dataset
