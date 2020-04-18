from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,LeakyReLU,\
                                    Flatten,Dense,Reshape,Conv2DTranspose,\
                                    Input, Activation, BatchNormalization, Layer,\
                                    Dropout

from core.utils import *
from ddsp.core import midi_to_hz
from ddsp.spectral_ops import F0_RANGE, LD_RANGE

class CPPN_f0(Model):

    def __init__(self,
                 n_nodes = 32,
                 n_hidden = 3,
                 activation = 'tanh',
                 t_scale=1,
                 z_scale=0.1,
                 latent_dim=16,
                 second_sig=False):

        super().__init__()
        self.t_scale = t_scale
        self.z_scale = z_scale
        self.latent_dim = latent_dim

        weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
        # weight_init = tf.keras.initializers.Ones()

        # input layers
        self.time_input = Dense(n_nodes,
                         input_shape=(1000, 1),
                         kernel_initializer=weight_init,
                         use_bias=False)

        self.z_input = Dense(n_nodes, input_shape=(1000, self.latent_dim))

        # fc model
        self.fc_model = Sequential()
        for i in range(n_hidden):
            self.fc_model.add(BatchNormalization())
            self.fc_model.add(Activation(activation))
            self.fc_model.add(Dense(n_nodes, kernel_initializer=weight_init))
        self.fc_model.add(Dense(1, kernel_initializer=weight_init))
        self.fc_model.add(BatchNormalization())
        self.fc_model.add(Activation("sigmoid"))
        if second_sig: self.fc_model.add(Activation("sigmoid"))

    def call(self,latent):
        # z = self.z_scale * tf.random.uniform((1, self.latent_dim),minval=-1.0, maxval=1.0) # (1, latent_dim)
        z = self.z_scale * latent # (1, latent_dim)
        z = tf.linalg.matmul(tf.ones((1000,1)), z) # (1000, zdim)
        Uz = self.z_input(z)

        t = self.t_scale * tf.reshape(tf.range(-1,1,delta=(1/500), dtype='float32'), (1,1000,1))
        Ut = self.time_input(t)

        U = Ut + Uz
        return self.fc_model(U)

class LatentGenerator(Layer):

    DEFAULT_ARGS = {
        "latent_dim": 100,

        # f0
        "f0_hidden_activation": 'tanh',
        "f0_t_scale": 0.5,
        "f0_z_scale": 0.1,
        "f0_second_sig": False,
        "f0_n_nodes":5,
        "f0_n_hidden":3,

        # ld
        "ld_hidden_activation": 'tanh',
        "ld_t_scale": 0.5,
        "ld_z_scale": 0.1,
        "ld_second_sig": True,
        "ld_n_nodes":3,
        "ld_n_hidden":1,

        # z
        "num_z_filters": 16,
        "dropout": False,
        "drop_rate": 0.5,

    }
    # TODO: only does one at a time
    def __init__(self, name="LatentGenerator", **kwargs):
        super().__init__(name=name)
        self.params = merge(self.DEFAULT_ARGS, kwargs)
        self.output_splits = (('f0_scaled', 1),('ld_scaled', 1),('z', 8))

        #define layers
        self.f0_cppn = CPPN_f0(n_nodes=self.params["f0_n_nodes"],
                               n_hidden=self.params["f0_n_hidden"],
                               t_scale=self.params["f0_t_scale"],
                               z_scale=self.params["f0_z_scale"],
                               latent_dim=self.params["latent_dim"],
                               activation=self.params["f0_hidden_activation"],
                               second_sig=self.params["ld_second_sig"])

        self.ld_cppn = CPPN_f0(n_nodes=self.params["ld_n_nodes"],
                               n_hidden=self.params["ld_n_hidden"],
                               t_scale=self.params["ld_t_scale"],
                               z_scale=self.params["ld_z_scale"],
                               latent_dim=self.params["latent_dim"],
                               activation=self.params["ld_hidden_activation"],
                               second_sig=self.params["ld_second_sig"])

        self.z_upsampler = self.build_z_upsampler(latent_dim=self.params["latent_dim"],
                                                  dropout=self.params["dropout"],
                                                  drop_rate=self.params["drop_rate"],)

    def call(self,inputs):
        """Generates outputs with dictionary of f0_scaled and ld_scaled."""
        latent = tf.random.normal((1,self.params["latent_dim"])) # (1, 100ish)
        z = self.z_upsampler(latent) # (1, 8, 1000, 1)
        z = tf.squeeze(z,axis=0)# (8, 1000, 1)

        f0 = self.f0_cppn(latent)# (1, 1000, 1)
        ld = self.ld_cppn(latent)# (1, 1000, 1)
        flz = tf.concat((f0,ld,z), axis=0)# (10, 1000, 1)

        # convert to dictionary
        flz = tf.transpose(flz) # (1, 1000, 10)
        outputs = ddsp.training.nn.split_to_dict(flz, self.output_splits)
        return outputs

    def gen_from_latent(self, inputs, latent):
        """Generates outputs with dictionary of f0_scaled and ld_scaled from vec of size (1, latent_dim)."""
        z = self.z_upsampler(latent) # (1, 8, 1000, 1)
        z = tf.squeeze(z,axis=0)# (8, 1000, 1)

        f0 = self.f0_cppn(latent)# (1, 1000, 1)
        ld = self.ld_cppn(latent)# (1, 1000, 1)
        flz = tf.concat((f0,ld,z), axis=0)# (10, 1000, 1)

        # convert to dictionary
        flz = tf.transpose(flz) # (1, 1000, 10)
        outputs = ddsp.training.nn.split_to_dict(flz, self.output_splits)
        return outputs

    def build_z_upsampler(self, latent_dim, dropout=False, drop_rate=0.5):
        # define the generator model
        generator = Sequential()
        # foundation for 1 x 125 signal
        n_nodes = 125 * 1 * self.params["num_z_filters"]
        generator.add(Dense(n_nodes, input_dim=latent_dim, dtype='float32'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Reshape((1, 125, self.params["num_z_filters"])))
        # upsample to 2 x 250
        generator.add(Conv2DTranspose(self.params["num_z_filters"], (3,3), strides=(2,2), padding='same'))
        generator.add(LeakyReLU(alpha=0.2))
        # convolution
        generator.add(Conv2D(self.params["num_z_filters"], (2,2), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        if dropout: generator.add(Dropout(drop_rate))
        # convolution
        generator.add(Conv2D(self.params["num_z_filters"], (2,2), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        if dropout: generator.add(Dropout(drop_rate))
        # upsample to 4 x 500
        generator.add(Conv2DTranspose(self.params["num_z_filters"], (3,3), strides=(2,2), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        # convolution
        generator.add(Conv2D(self.params["num_z_filters"], (3,3), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        if dropout: generator.add(Dropout(drop_rate))
        # convolution
        generator.add(Conv2D(self.params["num_z_filters"], (3,3), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        if dropout: generator.add(Dropout(drop_rate))
        # convolution
        generator.add(Conv2D(self.params["num_z_filters"], (3,3), dilation_rate=2, padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        if dropout: generator.add(Dropout(drop_rate))
        # upsample to 8 x 1000
        generator.add(Conv2DTranspose(self.params["num_z_filters"], (3,3), strides=(2,2), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        # convolution
        generator.add(Conv2D(self.params["num_z_filters"], (3,3), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        if dropout: generator.add(Dropout(drop_rate))
        # convolution
        generator.add(Conv2D(self.params["num_z_filters"], (3,3), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        if dropout: generator.add(Dropout(drop_rate))
        # convolution
        generator.add(Conv2D(self.params["num_z_filters"], (3,3), dilation_rate=2, padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        # convolve down to 1 filter
        generator.add(Conv2D(1, (3,3), activation='sigmoid', padding='same'))
        return generator

class UnPreprocessor(ddsp.training.preprocessing.Preprocessor):
    """Get (f0_hz and loudness_db) from (f0_scaled and ld_scaled)"""
    def __init__(self, time_steps=1000):
        super().__init__()
        self.time_steps = time_steps

    def __call__(self, features):
        super().__call__(features)
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

class Generator(Layer):

    DEFAULT_ARGS = {
        # Decoder
        "rnn_type":"gru",

        # Processor Group ### Still needs to be implemented
        "main_synth":"additive",

    }

    def __init__(self,name='generator',**kwargs):
        super().__init__(name=name)

        self.params = merge(self.DEFAULT_ARGS, kwargs)
        if self.params["main_synth"] == "additive":
            self.tambre_vec_name = "harmonic_distribution"
        elif self.params["main_synth"] == "wavetable":
            self.tambre_vec_name = "wavetables"

        self.latent_generator = LatentGenerator(**kwargs)
        self.unprocessor = UnPreprocessor(time_steps=1000)
        self.decoder = self.buildDecoder()
        self.processor_group = self.buildProcessorGroup()

    def call(self,inputs, label=0):
        generated = self.latent_generator(None) # no inputs. generating
        un_processed = self.unprocessor(generated)
        decoded = self.decoder(un_processed)
        sample = decoded
        sample['audio'] = self.processor_group(decoded)
        return sample

    def generate(self,label=0):
        """
            audio: (64000,)
            f0_hz: (1000,)
            loudness_db: (1000,)
            label: (1,)
        """
        sample = self.call(None)
        squeezedSample = {}
        useKeys = ["audio","f0_hz","loudness_db"]
        for key in useKeys:
            squeezedSample[key] = tf.squeeze(sample[key])
        squeezedSample["label"] = tf.convert_to_tensor([float(label)])
        return squeezedSample

    def gen_from_latent(self, latent):
        ''' Generates audio from given latent vector '''
        upsampled = self.latent_generator.gen_from_latent(None, latent)
        un_processed = self.unprocessor(upsampled)
        decoded = self.decoder(un_processed)
        sample = decoded
        sample['audio'] = self.processor_group(decoded)
        return sample

    def generate_batch(self, label=0, batch_size=8):
        """
            "audio": (batch_size, 64000),
            "f0_hz": (batch_size, 1000),
            "loudness_db": (batch_size, 1000)
            "label": (batch_size, 1)
        """
        samples = {"audio":[], "f0_hz": [], "loudness_db": [], "label": []}
        for i in range(batch_size):
            sample = self.generate(label=label)
            for key in samples.keys():
                samples[key].append(sample[key])

        for key in samples.keys():
            samples[key] = tf.convert_to_tensor(samples[key])

        return samples

    def buildDecoder(self):
        # rnn decoder .. TODO: figure out what this is!!
        decoder = ddsp.training.decoders.ZRnnFcDecoder(
            rnn_channels = 256,
            rnn_type = self.params["rnn_type"],
            ch = 256,
            layers_per_stack = 1,
            output_splits = (('amps', 1),
                             (self.tambre_vec_name, 45),
                             ('noise_magnitudes', 45)))
        return decoder

    def buildProcessorGroup(self):
        """
            Create actual synth structure (highly customizable)
        """
        if self.params['main_synth'] == "additive":
            main = ddsp.synths.Additive(n_samples=DEFAULT_N_SAMPLES,
                                            sample_rate=DEFAULT_SAMPLE_RATE, # this is defined in my_ddsp_utils
                                            name='main')

        if self.params['main_synth'] == "wavetable":
            main = ddsp.synths.Wavetable(n_samples=DEFAULT_N_SAMPLES,
                                            sample_rate=DEFAULT_SAMPLE_RATE, # this is defined in my_ddsp_utils
                                            name='main')

        noise = ddsp.synths.FilteredNoise(window_size=0,
                                          initial_bias=-10.0,
                                          name='noise')
        add = ddsp.processors.Add(name='add')
        reverb = ddsp.effects.Reverb(name='reverb', trainable=True)

        # package them together
        dag = [(main, ['amps', self.tambre_vec_name, 'f0_hz']),
               (noise, ['noise_magnitudes']),
               (add, ['noise/signal', 'main/signal']),
               (reverb, ['add/signal'])]
        processor_group = ddsp.processors.ProcessorGroup(dag=dag,
                                                         name='processor_group')

        return processor_group

    def showSummery(self):
        print("# params: ",self.count_params())
        print()
