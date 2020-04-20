from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error

from core.utils import *

class Discriminator(Model):
    DEFAULT_ARGS = {
        'batch_size': 8,
        'loss': binary_crossentropy,
    }

    def __init__(self, **kwargs):
        super().__init__(name='discriminator')
        self.params = merge(self.DEFAULT_ARGS, kwargs)
        self.preprocessor = self.buildPreprocessor()
        self.flzEncoder = self.buildFLZEncoder()
        self.classifier = self.buildClassifier()

    def call(self, sample, add_losses=True):
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
        classification = self.classifier(encoded_concat)

        if add_losses:
            self.add_loss(self.params["loss"](sample['label'],classification))
        return classification

    def buildPreprocessor(self):
        return ddsp.training.preprocessing.DefaultPreprocessor(time_steps=1000)

    def buildFLZEncoder(self):
        # TODO: try giving this an f0 encoder, like in ae_abs.gin
        encoder = ddsp.training.encoders.MfccTimeDistributedRnnEncoder(z_dims=6,
                                                                       z_time_steps=1000)
        return encoder

    def buildClassifier(self):
        #TODO
        # now encode even further down to a binary classification real or fake
        ''' Trip's Discriminator
        discriminator = Sequential()
        discriminator.add(InputLayer(((1000,8,1)), batch_size=self.params["batch_size"]))
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
        '''

        #''' New Discriminator
        discriminator = Sequential()
        discriminator.add(InputLayer(((1000,8,1)), batch_size=self.params["batch_size"]))
        # downsample to 500x3
        discriminator.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        discriminator.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        discriminator.add(Conv2D(16, (1,1), padding='same', activation='relu'))
        discriminator.add(BatchNormalization())
        discriminator.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # downsample to 250 x 2
        discriminator.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        discriminator.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        discriminator.add(Conv2D(16, (1,1), padding='same', activation='relu'))
        discriminator.add(BatchNormalization())
        discriminator.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # downsample to 125 x 1
        discriminator.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        discriminator.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        discriminator.add(Conv2D(8, (1,1), padding='same', activation='relu'))
        discriminator.add(BatchNormalization())
        discriminator.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # classify
        discriminator.add(Flatten())
        discriminator.add(Dense(100, activation='relu'))
        discriminator.add(Dense(1, activation='sigmoid'))
#         discriminator.summary()

        return discriminator
