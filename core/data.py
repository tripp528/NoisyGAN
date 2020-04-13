# local imports
from .utils import *

class FromTFRecords(ddsp.training.data.TFRecordProvider):
    """ class for getting dataset from tfrecords (for GAN) """
    def __init__(self,filepattern):
        super().__init__(filepattern)

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

class FromNSynth(ddsp.training.data.TfdsProvider):
  def __init__(self,
               name='nsynth/gansynth_subset.f0_and_loudness:2.3.0',
               split='train',
               data_dir='gs://tfds-data/datasets'):
    if data_dir == 'gs://tfds-data/datasets':
      logging.warning('If not on colab, this is very slow. Use data_dir param.')
    super().__init__(name, split, data_dir)

  def get_dataset(self, shuffle=True):
    """Returns dataset with slight restructuring of feature dictionary."""
    def preprocess_ex(ex):
        return {
            'audio': ex['audio'],
            'f0_hz': ex['f0']['hz'],
            'loudness_db': ex['loudness']['db'],
            'label': tf.convert_to_tensor([1.0]),
        }
    dataset = super().get_dataset(shuffle)
    dataset = dataset.map(preprocess_ex, num_parallel_calls=ddsp.training.data._AUTOTUNE)
    return dataset

# ------------------- iterator functions for GAN -------------------

class CombinedIter():
    def __init__(self, gen, data_provider, batch_size=8, overfit=False, shuffle=True):
        self.half_batch = int(batch_size/2)
        self.shapes = {"f0_hz": (1000,),
                  "loudness_db": (1000,),
                  "audio": (64000,),
                  "label": (1,)}
        if overfit:
            self.dataset = data_provider.get_batch(self.half_batch, shuffle=shuffle).take(self.half_batch).repeat()
        else:
            self.dataset = data_provider.get_batch(self.half_batch, shuffle=shuffle)
        self.gen = gen

    def getNext(self):
        generated = self.gen.generate_batch(label=0,batch_size=self.half_batch)
        real = next(iter(self.dataset))
        batch = {}
        for key in self.shapes:
            batch[key] = tf.concat([generated[key],real[key]],axis=0)

        return batch
