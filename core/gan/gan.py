from tensorflow.keras import Model

from core.utils import *
from .discriminator import Discriminator
from .generator import Generator

def compute_diversity(input_tensor):
    std_accross_batch_per_step = tf.math.reduce_std(input_tensor, axis=0)
    return tf.reduce_mean(std_accross_batch_per_step)

class GAN(Model):

    DEFAULT_ARGS = {
        'batch_size': 8,
        'loss': tf.keras.losses.binary_crossentropy,
        'add_diversity_loss': False,
    }

    def __init__(self, **kwargs):
        self.params = merge(self.DEFAULT_ARGS, kwargs)
        super().__init__(name='gan_model')
        self.gen = Generator(**self.params)
        self.disc = Discriminator(**kwargs)

    def call(self, inputs=None):
        generated = self.gen.generate_batch(label=1,batch_size=self.params["batch_size"])
        classification = self.disc(generated, add_losses=False)
        self.add_loss(self.params["loss"](generated["label"], classification))

        if self.params["add_diversity_loss"]:
            self.add_loss(compute_diversity(generated))

        return classification
