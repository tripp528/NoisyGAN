from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,LeakyReLU,\
                                    Flatten,Dense,Reshape,Conv2DTranspose,\
                                    Input
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam

from core.utils import *
from ddsp.core import midi_to_hz
from ddsp.spectral_ops import F0_RANGE, LD_RANGE

class UnPreprocessor(ddsp.training.preprocessing.Preprocessor):
    """Get (f0_hz and loudness_db) from (f0_scaled and ld_scaled)"""
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
                 output_splits=(('f0_scaled', 1),
                                ('ld_scaled', 1),
                                ('z', 6)), # Changed from 8
                 name="LatentGenerator"):
        super().__init__(name=name)
        self.latent_dim = latent_dim

        #define layers
        self.f0_cppn = self.build_f0_cppn(n_nodes = 100, n_hidden = 2, activation = 'relu')
        self.ld_cppn = self.build_ld_cppn(n_nodes = 100, n_hidden = 2, activation = 'relu')
        self.upsampler = self.build_z_upsampler()

        self.output_splits = output_splits
        self.n_out = sum([v[1] for v in output_splits])

    def call(self,inputs):
        """Generates outputs with dictionary of f0_scaled and ld_scaled."""
        latent = self.generate_latent_point()       # (1, 100) ( ish )
        upsampled = self.upsampler(latent)        # (1, 8, 1000, 1)
        upsampled = tf.squeeze(upsampled,axis=0)# (8, 1000, 1)

        f0_latent, ld_latent, z_latent = tf.split(upsampled, [1,1,6] , axis = 0)

        # test messing with f0 and ld
        f0_input = ((np.arange(1000, dtype='float32') / (1000 - 1)) - 0.5).reshape(1,1000,1)
        #f0_input = tf.reshape(tf.range(0,1,delta=(1/1000), dtype='float32'), (1,1000,1))

        ###f0_latent = tf.math.add(f0_latent, f0_input)
        f0 = self.f0_cppn(f0_input) # (1, 1000, 1)

        #ld = self.ld_cppn(np.ones(1000).reshape(1,1000,1) * .7)
        #ld_input = ((np.arange(1000, dtype='float32') / (1000 - 1)) - 0.5).reshape(1,1000,1)
        ld_input = tf.reshape(tf.range(0,1,delta=(1/1000), dtype='float32'), (1,1000,1))
        ###ld_latent = tf.math.add(ld_latent, ld_input)
        ld = self.ld_cppn(ld_input)

        #ldf0 = tf.convert_to_tensor(np.float32(np.concatenate([f0,ld])))
        ldf0 = tf.concat((f0,ld), axis=0)
        upsampled = tf.concat((ldf0, z_latent),axis=0)
        x = tf.transpose(upsampled)                 # (1, 1000, 8)

        # convert to dictionary
        outputs = ddsp.training.nn.split_to_dict(x, self.output_splits)
        return outputs

    def generate_latent_point(self):
        # generate points in the latent space
        latent = np.random.randn(self.latent_dim)
        latent = latent.reshape(1, self.latent_dim)
        # lt = (np.arange(self.latent_dim) / (self.latent_dim - 1)) - 0.5
        # latent = lt.reshape(1, self.latent_dim)
        return tf.convert_to_tensor(latent)

    def build_z_upsampler(self):
        # define the generator model
        generator = Sequential()
        # foundation for 1 x 125 signal
        n_nodes = 125 * 1 * 16
        generator.add(Dense(n_nodes, input_dim=self.latent_dim, dtype='float32'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Reshape((1, 125, 16)))
        # upsample to 2 x 250
        generator.add(Conv2DTranspose(16, (3,3), strides=(2,2), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        # upsample to 4 x 500
        generator.add(Conv2DTranspose(16, (3,3), strides=(2,2), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Conv2D(1, (3,3), activation='sigmoid', padding='same'))
        # upsample to 8 x 1000
        generator.add(Conv2DTranspose(16, (3,3), strides=(2,2), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Conv2D(1, (3,3), activation='sigmoid', padding='same'))

        return generator

    def build_f0_cppn(self, n_nodes = 100, n_hidden = 2, activation = 'relu'):
        ''' MLP, takes 1 input, returns 1 output '''
        input = Input(shape=(None,1), dtype='float32')
        x = Dense(n_nodes, activation = activation, name = "f0_cppn_input")(input)

        for i in range(n_hidden-1):
            x = Dense(n_nodes, activation = activation, name = "f0_cppn_dense_%04d" % i)(x)

        out = Dense(1, activation='sigmoid', name = "f0_cppn_output")(x)

        cppn = Model(input, out)

        return cppn

    def build_ld_cppn(self, n_nodes = 100, n_hidden = 2, activation = 'relu'):
        ''' MLP, takes 1 input, returns 1 output '''
        input = Input(shape=(None,1), dtype='float32')
        x = Dense(n_nodes, activation = activation, name = "ld_cppn_input")(input)

        for i in range(n_hidden-1):
            x = Dense(n_nodes, activation = activation, name = "ld_cppn_dense_%04d" % i)(x)

        out = Dense(1, activation='sigmoid', name = "ld_cppn_output")(x)

        cppn = Model(input, out)

        return cppn
        '''
        cppn = Sequential()
        cppn.add(Dense(n_nodes, input_dim=1, dtype='float32', activation=activation))

        # Add ddense layers
        for i in range(n_hidden):
            cppn.add(Dense(n_nodes, activation=activation))

        cppn.add(Dense(1, activation='sigmoid'))

        return cppn
        '''





class Generator(tf.keras.layers.Layer):
    def __init__(self,name='generator',latent_dim=100):
        super().__init__(name=name) #add Generator??
        self.latent_dim = latent_dim

        self.latent_generator = LatentGenerator(latent_dim=latent_dim)
        self.unprocessor = self.buildUnprocessor()
        self.decoder = self.buildDecoder()
        self.processor_group = self.buildProcessorGroup()

        # build the model and display summary
        # self.call(None)
        # self.build(None)
        # self.showSummery()

    def generate(self,label=0):
        """returns {
            audio: (64000,)
            f0_hz: (1000,)
            loudness_db: (1000,)
            label: (1,)
        }"""
        sample = self.call(None)
        squeezedSample = {}
        useKeys = ["audio","f0_hz","loudness_db"]
        for key in useKeys:
            squeezedSample[key] = tf.squeeze(sample[key])
        squeezedSample["label"] = tf.convert_to_tensor([float(label)])
        return squeezedSample

    def generate_batch(self, label=0, batch_size=8):
        """returns {

            "audio": (batch_size, 64000),
            "f0_hz": (batch_size, 1000),
            "loudness_db": (batch_size, 1000)
            "label": (batch_size, 1)

        }"""
        samples = {"audio":[], "f0_hz": [], "loudness_db": [], "label": []}
        for i in range(batch_size):
            sample = self.generate(label=label)
            for key in samples.keys():
                samples[key].append(sample[key])

        for key in samples.keys():
            samples[key] = tf.convert_to_tensor(samples[key])

        return samples

    def call(self,inputs, label=0):
        generated = self.latent_generator(None) # no inputs. generating
        un_processed = self.unprocessor(generated)
        decoded = self.decoder(un_processed)
        sample = decoded
        sample['audio'] = self.processor_group(decoded)
        return sample

    def buildUnprocessor(self):
        # Default preprocessor that resamples features and adds `f0_hz` key.
        unprocessor = UnPreprocessor(time_steps=1000)
        return unprocessor

    def buildDecoder(self):
        # rnn decoder .. TODO: figure out what this is!!
        decoder = ddsp.training.decoders.ZRnnFcDecoder(
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
