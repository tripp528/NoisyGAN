from core.utils import *
from .discriminator import binary_crossentropy

class GAN(ddsp.training.models.Model):
    def __init__(self, gen, disc, losses=[binary_crossentropy()]):
        super().__init__(name='gan_model', losses=losses)
        self.gen = gen
        self.disc = disc

    def call(self, inputs, batch_size=8): #inputs are NONE
        # label and training param don't matter here - we add our own label and loss
        generated = self.gen.generate_batch(batch_size=batch_size)
        classification = self.disc(generated)
        label = tf.convert_to_tensor([1]) #trying to trick the frozen discriminator
        self.add_losses(label, classification)

        return classification
