import pickle

from ddsp_autoencoder import *

# preprocess into tfrecords
input_audio_filepattern = "./Data/piano/piano30s.wav"
output_tfrecord_path = './Data/piano/piano30s.tfrecord'
dataset = DDSP_DATASET(input_audio_filepattern, output_tfrecord_path,buildRecords=False)

#build model
autoencoder = DDSP_AUTOENCODER(dataset)
autoencoder.train()

audio, audio_gen = autoencoder.predict(sampleNum=3)
