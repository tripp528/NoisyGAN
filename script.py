import pickle

from absl import logging
import tensorflow as tf

from ddsp_autoencoder import *

# get flags from command line
flags = tf.compat.v1.flags
flags.DEFINE_boolean("build_records",False,"build tfrecords or use existing ones in model_dir")
flags.DEFINE_string("model_dir","./models/auto/","model directory")
flags.DEFINE_integer("iters", 30, "number iterations to train model")
opt = flags.FLAGS

# preprocess into tfrecords
input_audio_filepattern = "./Data/piano/piano30s.wav"
output_tfrecord_path = './Data/piano/piano30s.tfrecord'
dataset = DDSP_DATASET(input_audio_filepattern, output_tfrecord_path,buildRecords=opt.build_records)

#build model
logging.set_verbosity(logging.INFO)
autoencoder = DDSP_AUTOENCODER(dataset,model_dir=opt.model_dir)
autoencoder.train(iterations=opt.iters)
