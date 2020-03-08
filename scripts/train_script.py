import pickle

from absl import logging

from ddsp_autoencoder3 import *

# get flags from command line
flags = tf.compat.v1.flags
# flags.DEFINE_boolean("build_records",False,"build tfrecords or use existing ones in model_dir")
flags.DEFINE_string("model_dir","./models/auto/","model directory")
flags.DEFINE_string("audio_input",None,"audio file")
flags.DEFINE_string("tfrecord_pattern",'./Data/piano/piano30s.tfrecord',"where to put the tfrecords")
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

#build model
logging.set_verbosity(logging.INFO)
train_set = DDSP_DATASET(opt.tfrecord_pattern, audio_input=opt.audio_input)
model = Solo_Autoencoder()
trainer = DDSP_TRAINER(model,model_dir=opt.model_dir,gpus=opt.gpus)
autoencoder.train(train,iterations=opt.iters)
