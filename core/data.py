# local imports
from .utils import *

class FromTFRecords(ddsp.training.data.TFRecordProvider):
    """ class for getting dataset from tfrecords (for GAN) """
    def __init__(self,filepattern,label=1):
        super().__init__(filepattern)
        self.label = label

    def get_dataset(self, shuffle=True):
        def parse_tfexample(record):
            example = tf.io.parse_single_example(record, self.features_dict)
            del example["f0_confidence"]

            example["label"] = tf.convert_to_tensor([1.0])
            return example

        filenames = tf.data.Dataset.list_files(self._file_pattern, shuffle=shuffle)
        dataset = filenames.interleave(
            map_func=tf.data.TFRecordDataset,
            cycle_length=40,
            num_parallel_calls=ddsp.training.data._AUTOTUNE)

        dataset = dataset.map(parse_tfexample, num_parallel_calls=ddsp.training.data._AUTOTUNE)
        return dataset

    def getAudio(self,sampleNum=0):
        samples = list(iter(self.get_dataset(shuffle=False)))
        sample = samples[sampleNum]
        audio = sample["audio"]
        return audio

    def getSample(self,sampleNum=0):
        samples = list(iter(self.get_dataset(shuffle=False)))
        howmany = len(samples)
        sample = samples[sampleNum % howmany]
        # for key in sample.keys():
        #     sample[key] = np.expand_dims(sample[key],axis=[0])
        return sample

    def get_n_samples(self):
        return self.getAudio().shape[0]

# ------------------- iterator functions for GAN -------------------

def combined_sample_iter(gen, data_provider, batch_size=8):
    half_batch = int(batch_size/2)
    shapes = {"f0_hz": (1000,),
                    "loudness_db": (1000,),
                    "audio": (64000,),
                    "label": (1,)}

    i = -1
    while True:
        i+=1
        # logging.info("generating... batch "+str(i))
        logging.info("generating...")
        generated = gen.generate_batch(label=0,batch_size=half_batch)
        real = next(iter(data_provider.get_batch(half_batch, shuffle=True, repeats=-1)))
        batch = {}
        for key in shapes:
            batch[key] = tf.concat([generated[key],real[key]],axis=0)

        yield batch

def real_sample_iter(data_provider, batch_size=8):
    i = -1
    while True:
        i+=1
        real = next(iter(data_provider.get_batch(batch_size, shuffle=True, repeats=-1)))
        yield real

def fake_sample_iter(gen, batch_size=8):
    shapes = {"f0_hz": (1000,),
                    "loudness_db": (1000,),
                    "audio": (64000,),
                    "label": (1,)}

    i = -1
    while True:
        i+=1
        logging.info("generating...")
        generated = gen.generate_batch(label=0,batch_size=batch_size)
        batch = {}
        for key in shapes:
            batch[key] = generated[key]

        yield batch
