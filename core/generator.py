from .ddsp_dataset import *


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,BatchNormalization,LeakyReLU,\
                                    Flatten,Dense,Reshape,Conv2DTranspose
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam


from ddsp.core import midi_to_hz
from ddsp.spectral_ops import F0_RANGE, LD_RANGE

class UnPreprocessor(ddsp.training.preprocessing.Preprocessor):

    def __init__(self, time_steps=1000):
        super().__init__()
        self.time_steps = time_steps

    def __call__(self, features, training=True):
        super().__call__(features, training)
        return self._un_processing(features)

    def _un_processing(self, features):
        # features['f0_scaled'] = hz_to_midi(features['f0_hz']) / F0_RANGE
        # to scale, 1. hztomidi, 2. divide by f0range
        # to unscale, 1. * f0range, 2. miditohz
        features['f0_hz'] = midi_to_hz( features['f0_scaled'] * F0_RANGE )

        # features['ld_scaled'] = (features['loudness_db'] / LD_RANGE) + 1.0
        # to scale, 1. / ldrange, 2. + 1
        # to unscale, 1. -1, 2. * ldrange
        features['loudness_db'] = (features['ld_scaled'] - 1) * LD_RANGE
        return features


class LatentGenerator(tf.keras.layers.Layer):
    # TODO: only does one at a time
    def __init__(self,
                 latent_dim=100,
                 output_splits=(('f0_scaled', 1), ('ld_scaled', 1)),
                 name="LatentGenerator"):
        super().__init__(name=name)
        self.latent_dim = latent_dim

        #define layers
        self.upsampler = self.buildUpsampler()

        self.output_splits = output_splits
        self.n_out = sum([v[1] for v in output_splits])

    def call(self,inputs):
        """Generates outputs with dictionary of f0_scaled and ld_scaled."""
        latent = self.generate_latent_point()
        upsampled = self.upsampler(latent) # (1, 2, 1000, 1)
        # instead of squeezing make a for loop here...
        upsampled = np.squeeze(upsampled,axis=0) # (2, 1000, 1)
        f, l = upsampled[0], upsampled[1]
        f = tf.convert_to_tensor(np.expand_dims(f,axis=0))
        l = tf.convert_to_tensor(np.expand_dims(l,axis=0))
        x = tf.concat([f, l], axis=-1)
        outputs = ddsp.training.nn.split_to_dict(x, self.output_splits)
        return outputs

    def generate_latent_point(self):
        # generate points in the latent space
        latent = np.random.randn(self.latent_dim)
        # reshape into a batch of inputs for the network
        latent = latent.reshape(1, self.latent_dim)
        return latent

    def buildUpsampler(self):
        # define the generator model
        generator = Sequential()
        # foundation for 1 x 250 signal
        n_nodes = 250 * 1 * 16
        generator.add(Dense(n_nodes, input_dim=self.latent_dim, dtype='float32'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Reshape(( 1,250, 16)))
        # upsample to 1 x 500
        generator.add(Conv2DTranspose(16, (3,3), strides=(1,2), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        # upsample to 2 x 1000
        generator.add(Conv2DTranspose(16, (3,3), strides=(2,2), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Conv2D(1, (3,3), activation='sigmoid', padding='same'))

#         generator.summary()
        return generator


class Generator(tf.keras.layers.Layer):
    def __init__(self,name='generator',latent_dim=100):
        super().__init__(name=name)
        self.latent_dim = latent_dim

        self.latent_generator = LatentGenerator(latent_dim=latent_dim)
        self.unprocessor = self.buildUnprocessor()
        self.decoder = self.buildDecoder()
        self.processor_group = self.buildProcessorGroup()

        # build the model and display summary
        self.call(None)
        self.build(None)
        self.showSummery()

    def generate(self):
        return self.call(None)

    def call(self,inputs):
        generated = self.latent_generator(None) # no inputs. generating
        un_processed = self.unprocessor(generated)
        decoded = self.decoder(un_processed)
        sample = decoded
        sample['audio'] = tf.squeeze(self.processor_group(decoded))
        return sample

    def buildUnprocessor(self):
        # Default preprocessor that resamples features and adds `f0_hz` key.
        unprocessor = UnPreprocessor(time_steps=1000)
        return unprocessor

    def buildDecoder(self):
        # rnn decoder .. TODO: figure out what this is!!
        decoder = ddsp.training.decoders.RnnFcDecoder(
            rnn_channels = 256,
            rnn_type = 'gru',
            ch = 256,
            layers_per_stack = 1,
            output_splits = (('amps', 1),
                             ('harmonic_distribution', 45),
                             ('noise_magnitudes', 45)))
        return decoder

    def buildProcessorGroup(self):
        """ Create actual synth structure (highly customizable)

        Defaults to n_samples and sample_rate defined in my_ddsp_utils

        """
        # Create Processors.
        additive = ddsp.synths.Additive(n_samples=DEFAULT_N_SAMPLES,
                                        sample_rate=DEFAULT_SAMPLE_RATE, # this is defined in my_ddsp_utils
                                        name='additive')
        noise = ddsp.synths.FilteredNoise(window_size=0,
                                          initial_bias=-10.0,
                                          name='noise')
        add = ddsp.processors.Add(name='add')
        reverb = ddsp.effects.Reverb(name='reverb', trainable=True)

        # package them together
        dag = [(additive, ['amps', 'harmonic_distribution', 'f0_hz']),
               (noise, ['noise_magnitudes']),
               (add, ['noise/signal', 'additive/signal']),
               (reverb, ['add/signal'])]
        processor_group = ddsp.processors.ProcessorGroup(dag=dag,
                                                         name='processor_group')

        return processor_group

    def showSummery(self):
        print("# params: ",self.count_params())
        print()
