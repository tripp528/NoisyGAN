import pickle

from ddsp_dataset import *
logging.set_verbosity(logging.INFO)

# get flags from command line
flags = tf.flags
# flags.DEFINE_boolean("build_records",False,"build tfrecords or use existing ones in model_dir")
# flags.DEFINE_string("model_dir","./models/auto/","model directory")
flags.DEFINE_string("audio_input","./Data/piano/piano30s.wav","audio file")
flags.DEFINE_string("record_pattern",'./Data/piano/piano30s.tfrecord',"where to put the tfrecords")
# flags.DEFINE_integer("iters", 30, "number iterations to train model")
flags.DEFINE_boolean("gpu_limit",False,"limit on gpu memory (for linux box")
opt = flags.FLAGS

# gpu limit
if opt.gpu_limit:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4624)])


# preprocess into tfrecords
dataset = DDSP_DATASET(opt.audio_input, opt.record_pattern,buildRecords=True)
