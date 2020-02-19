import glob
import time

import tensorflow.compat.v1 as tf
print("tf versionn: ",tf.__version__)
# tf.compat.v1.enable_v2_behavior()
import tensorflow_datasets as tfds

import ddsp
import ddsp.training
from ddsp.training.data_preparation.prepare_tfrecord_lib import prepare_tfrecord

# local imports
from my_ddsp_utils import *

def buildTFRecords(input_audio_filepatterns, output_tfrecord_path):
    train_file_pattern = output_tfrecord_path+"*"

    if not glob.glob(input_audio_filepatterns):
        raise ValueError('No audio files found. Please use the previous cell to '
                        'upload.')
    else:
        print("found", input_audio_filepatterns)

    input_audio_paths = []
    for filepattern in input_audio_filepatterns:
        input_audio_paths.extend(tf.io.gfile.glob(filepattern))

    prepare_tfrecord(
        input_audio_paths,
        output_tfrecord_path,
        num_shards=10,
        sample_rate=sample_rate)
