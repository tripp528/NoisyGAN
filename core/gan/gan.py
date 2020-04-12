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

    def call(self, inputs): #inputs are NONE
        # label and training param don't matter here - we add our own label and loss
        generated = self.gen.generate_batch(batch_size=self.params["batch_size"])
        classification = self.disc(generated)
        label = tf.convert_to_tensor([1]) #trying to trick the frozen discriminator
        self.add_loss(binary_crossentropy(label, classification))

        return classification
