from tensorflow.keras import Model
from tensorflow.keras.losses import binary_crossentropy

from core.utils import *
from .discriminator import Discriminator
from .generator import Generator

class GAN(Model):

    DEFAULT_ARGS = {
        'batch_size': 8
    }

    def __init__(self, **kwargs):
        self.params = merge(self.DEFAULT_ARGS, kwargs)
        super().__init__(name='gan_model')
        self.gen = Generator(**self.params)
        self.disc = Discriminator(batch_size=self.params["batch_size"])

    def call(self, inputs=None):
        generated = self.gen.generate_batch(label=1,batch_size=self.params["batch_size"])
        classification = self.disc(generated, add_losses=False)
        self.add_loss(binary_crossentropy(generated["label"], classification))
        return classification
