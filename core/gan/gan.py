from tensorflow.keras import Model

from core.utils import *
from .discriminator import Discriminator
from .generator import Generator

class GAN(Model):

    DEFAULT_ARGS = {
        'batch_size': 8,
        'loss': tf.keras.losses.binary_crossentropy,
        'diversity_loss_weight': 0,
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

        if self.params["diversity_loss_weight"] > 0:
            std_accross_batch_per_step = tf.math.reduce_std(generated["audio"], axis=0)
            diversity = tf.reduce_mean(std_accross_batch_per_step)
            diversity_loss = self.params["diversity_loss_weight"] / diversity
            diversity_loss = tf.repeat(diversity_loss, repeats=self.params["batch_size"])
            self.add_loss(diversity_loss)

        return classification
