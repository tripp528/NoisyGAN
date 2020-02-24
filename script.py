import pickle

from absl import logging

from ddsp_autoencoder import *

# get flags from command line
flags = tf.compat.v1.flags
flags.DEFINE_boolean("build_records",False,"build tfrecords or use existing ones in model_dir")
flags.DEFINE_string("model_dir","./models/auto/","model directory")
flags.DEFINE_string("audio_input","./Data/piano/piano30s.wav","audio file")
flags.DEFINE_string("record_pattern",'./Data/piano/piano30s.tfrecord',"where to put the tfrecords")
flags.DEFINE_integer("iters", 10000, "number iterations to train model")
flags.DEFINE_list("gpus", None, "list of gpu addresses if using multiple")
flags.DEFINE_boolean("gpu_limit",False,"limit on gpu memory (for linux box")
opt = flags.FLAGS

# see all logging messages
logging.set_verbosity(logging.INFO)

# gpu limit
if opt.gpu_limit:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4624)])

# preprocess into tfrecords
dataset = DDSP_DATASET(opt.audio_input, opt.record_pattern,buildRecords=opt.build_records)

#build model
logging.set_verbosity(logging.INFO)
autoencoder = DDSP_AUTOENCODER(dataset,model_dir=opt.model_dir,gpus=opt.gpus)
autoencoder.train(iterations=opt.iters)
