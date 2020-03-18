from .ddsp_dataset import *

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,BatchNormalization,LeakyReLU,\
                                    Flatten,Dense,Reshape,Conv2DTranspose,InputLayer
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam

class binary_crossentropy(tf.keras.layers.Layer):
    def __init__(self,name = "binary_crossentropy"):
        super().__init__(name=name)

    def call(self, target, output):
        return tf.keras.losses.binary_crossentropy(target,output)

class Discriminator(ddsp.training.models.Model):
    """for now, just re-encode the fake sample.
        in the future, just feed flz from fake right in
        TODO: subclass ddsp Model
    """
    def __init__(self, losses=[binary_crossentropy()], batch_size=32):
        super().__init__(name='discriminator', losses=losses)
        self.batch_size = batch_size
        self.preprocessor = self.buildPreprocessor()
        self.flzEncoder = self.buildFLZEncoder()
        self.classifier = self.buildClassifier()

    def call(self, sample, training=True):
        # audio key must have shape (1,64000)
        preprocessed = self.preprocessor(sample)

        # only get flz if label=1? ... try both
        encoded = self.flzEncoder(preprocessed)

        # shape the shit into [1, 1000, 8, 1]
        encoded_concat = tf.concat([encoded['f0_scaled'],
                                    encoded['ld_scaled'],
                                    encoded['z']], axis=2)
        encoded_concat = tf.expand_dims(encoded_concat,axis=3)

        # classify if it's real or not
        classification = self.classifier(encoded_concat, training=training)

        if training:
            label = sample['label']
            # print(label,classification)
            self.add_losses(label, classification)

        return classification

    def buildPreprocessor(self):
        return ddsp.training.preprocessing.DefaultPreprocessor(time_steps=1000)

    def buildFLZEncoder(self):
        # TODO: try giving this an f0 encoder, like in ae_abs.gin
        encoder = ddsp.training.encoders.MfccTimeDistributedRnnEncoder(z_dims=6,
                                                                       z_time_steps=1000)
        return encoder

    def buildClassifier(self,training=True):
        #TODO
        # now encode even further down to a binary classification real or fake
        discriminator = Sequential()
        discriminator.add(InputLayer(((1000,8,1)), batch_size=self.batch_size))
        # downsample to 500x3
        discriminator.add(Conv2D(16, (3,3), strides=(2, 2), padding='same'))
        discriminator.add(BatchNormalization())
        discriminator.add(LeakyReLU(alpha=0.2))
        # downsample to 250 x 2
        discriminator.add(Conv2D(16, (3,3), strides=(2, 2), padding='same'))
        discriminator.add(BatchNormalization())
        discriminator.add(LeakyReLU(alpha=0.2))
        # downsample to 125 x 1
        discriminator.add(Conv2D(16, (3,3), strides=(2, 2), padding='same'))
        discriminator.add(BatchNormalization())
        discriminator.add(LeakyReLU(alpha=0.2))
        # classify
        discriminator.add(Flatten())
        discriminator.add(Dense(1, activation='sigmoid'))
#         discriminator.summary()

        return discriminator
