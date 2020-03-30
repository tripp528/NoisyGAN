from core.utils import *
import gin

# @gin.configurable # I think we need this to use get_controls()... TODO: TEST THAT OUT
class Solo_Autoencoder(ddsp.training.models.Model):

    def __init__(self):
        """
        Set the layers for use in call()
            Read documentation of "keras model subclass API" to understand

        """

        # call init of ddsp.training.models.Model (which calls keras.Model.__init__)
        super().__init__(name='autoencoder', losses=self.buildLosses())

        self.preprocessor = self.buildPreprocessor()
        self.encoder = self.buildEncoder()
        self.decoder = self.buildDecoder()
        self.processor_group = self.buildProcessorGroup()

    def call(self, features, training=True):
        """ Run the core of the network, get predictions and loss.

        Layer is a loosly used term in the comments; it is typically refering to a
            group  of layers (every model is technically a layer in keras).

        Got rid of encode() and decode() functions.
            encode() passed through preprocessor and encoding
            decode() passed through decoder and processor group

            I decided to just put all four here. Makes it more generalizeable to
                non-Autoencoder models (like GAN).
        """

        # preprocessing layer
        preprocessed = self.preprocessor(features, training=training)

        # encoding layer
        encoded = preprocessed if self.encoder is None else self.encoder(preprocessed)

        # decoding layer
        decoded = self.decoder(encoded, training=training)
        processor_inputs = decoded

        # run it through the processor group (the synth & effects)
        audio_gen = self.processor_group(processor_inputs)

        # add losses (if training) for backpropagation
        if training:
            self.add_losses(features['audio'], audio_gen)
        return audio_gen

    def get_controls(self, features, keys=None, training=False):
        """Returns specific processor_group controls.

        Had to rewrite due to  getting rid encode and decode functions.

        TODO: Does this get called with this line?
            harmonic_distribution = get('additive/controls/harmonic_distribution')
        """
        # preprocessing layer
        preprocessed = self.preprocessor(features, training=training)

        # encoding layer
        encoded = preprocessed if self.encoder is None else self.encoder(preprocessed)

        # decoding layer
        decoded = self.decoder(encoded, training=training)
        processor_inputs = decoded

        # get the controls
        controls = self.processor_group.get_controls(processor_inputs)
        return controls if keys is None else {k: controls[k] for k in keys}


    def buildPreprocessor(self):
        # Default preprocessor that resamples features and adds `f0_hz` key.
        preprocessor = ddsp.training.preprocessing.DefaultPreprocessor(time_steps=1000)
        return preprocessor

    def buildEncoder(self):
        # solo instrument model : not training Z
        encoder = None

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

    def buildLosses(self):
        # Loss functions
        spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1',
                                                 mag_weight=1.0,
                                                 logmag_weight=1.0)
        losses=[spectral_loss]
        return losses
