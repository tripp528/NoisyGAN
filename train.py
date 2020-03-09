from core import *

# get flags from command line
flags = tf.compat.v1.flags
flags.DEFINE_string("model_dir","./models/auto/","model directory")
flags.DEFINE_string("audio_input",None,"audio file")
flags.DEFINE_string("tfrecord_pattern",'./Data/piano/piano30s.tfrecord',"where to put the tfrecords")
flags.DEFINE_integer("iters", 10, "number iterations to train model")
flags.DEFINE_list("gpus", None, "list of gpu addresses if using multiple")
flags.DEFINE_boolean("gpu_limit",False,"limit on gpu memory (for linux box")
opt = flags.FLAGS

# gpu limit
if opt.gpu_limit:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4624)])

#build model
train_set = DDSP_DATASET(opt.tfrecord_pattern, audio_input=opt.audio_input)
trainer = DDSP_TRAINER(model_dir=opt.model_dir,gpus=opt.gpus)
trainer.train(train_set,iterations=opt.iters)
