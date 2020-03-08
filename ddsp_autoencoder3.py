from ddsp_dataset import *
import gin

class DDSP_TRAINER(ddsp.training.train_util.Trainer):
    """

    Extension of Trainer, which is defined as:
        Class to bind an optimizer, model, strategy, and training step function.

    The following functionality is added:

    1) This class aims to give default arguments to the trainer, as well as packaging
        in a dataset object into the mix. The idea is to just be able to pass in the
        filepatterns of your tfrecords. This means that you pass a file pattern
        instead of a data_provider into train and predict functions.

            This is accomplished using the DDSP_DATASET class, which takes filepatterns
        and constructs datasets. It also supports passing in an audio_input
        filepattern, which will automatically construct the tfrecords in the
        tfrecord_filepattern location before constructing the dataset.

            By passing in dataset paths instead of objects, it eliminates the need for
        preprocessing in your notebooks and scripts.

    2) This class bundles in a model directory (model_dir) as well. In ddsp.train_util,
        this is not passed in until you actually call the train function (outside of
        Trainer class).

            This enables automatic resotoring of your model from any checkpoints already
        in the model directory.

    3) Takes gpu arguments directly, and constructs a strategy internally from that.
        This eliminates the need to construct a strategy in your notebook / script.

    4) Constructs a model internally, without outside definition. This means you don't
        need to use the model class directly.

        TODO: take in a string, and construct a different model depending on the string.


    """

    def __init__(self, tfrecord_pattern, model_dir, audio_input=None, restore=False, gpus=None):
        self.ddsp_dataset = DDSP_DATASET(tfrecord_pattern, audio_input=audio_input)
        self.model_dir = model_dir
        self.strategy = ddsp.training.train_util.get_strategy(gpus=gpus) # get distribution strategy (change if using gpus/tpus)
        self.buildModel() # sets self.model

        super().__init__(self.model,
                        self.strategy,
                        checkpoints_to_keep=10,
                        learning_rate=0.001,
                        lr_decay_steps=10000,
                        lr_decay_rate=0.98,
                        grad_clip_norm=3.0)

        # Build model, easiest to just run forward pass.
        dataset = self.distribute_dataset(self.ddsp_dataset.getDataset())
        self.build(next(iter(dataset)))

        # restore from checkpoint
        if restore:
            self.call_restore()

    def buildModel(self):
        with self.strategy.scope():
            # Put it together in a model.
            self.model = Solo_Autoencoder()

    def predict(self, sampleNum=0, tfrecord_pattern=None):
        # Run a batch of predictions.
        if tfrecord_pattern == None:
            dataset = self.ddsp_dataset
        else:
            dataset = DDSP_DATASET(tfrecord_pattern)

        sample = dataset.getSample(sampleNum=sampleNum)
        start_time = time.time()
        controls =  self.model.get_controls(sample)
        audio_gen = controls['processor_group']['signal']
        print('Prediction took %.1f seconds' % (time.time() - start_time))
        return sample["audio"], audio_gen

    def train(self, iterations=10000):
        ddsp.training.train_util.train(self.ddsp_dataset.data_provider,
              self,
              batch_size=32,
              num_steps=iterations,
              steps_per_summary=10,
              steps_per_save=10,
              model_dir=self.model_dir)
            # default stuff:
            # data_provider,
            #   trainer,
            #   batch_size=32,
            #   num_steps=1000000,
            #   steps_per_summary=300,
            #   steps_per_save=300,
            #   model_dir='~/tmp/ddsp'

    def call_restore(self):
        self.model_dir = find_model_dir(self.model_dir)
        ckpt = ddsp.training.train_util.get_latest_chekpoint(self.model_dir)
        logging.info("restoring... "+ckpt)
        self.restore(ckpt)

@gin.configurable# i think we need this to use get_controls... TODO: TEST THAT OUT
class Solo_Autoencoder(ddsp.training.models.Model):

    def __init__(self):
        """
        Set the layers for use in call()
            Read documentation of "keras model subclass API" to understand

        """

        self.preprocessor = self.buildPreprocessor()
        self.encoder = self.buildEncoder()
        self.decoder = self.buildDecoder()
        self.processor_group = self.buildProcessorGroup()

        # call init of ddsp.training.models.Model
        super().__init__(name='autoencoder2', losses=self.buildLosses())

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
        if self.encoder:
            encoded = self.encoder(preprocessed)
        else:
            encoded = preprocessed

        # decoding layer (gets back processor_group inputs)
        decoded = self.decoder(conditioning, training=training)

        # run it through the processor group (the synth & effects)
        audio_gen = self.processor_group(decoded)

        # add losses (if training) for backpropagation
        if training:
            self.add_losses(features['audio'], audio_gen)
        return audio_gen

    def get_controls(self, features, keys=None, training=False):
        """Returns specific processor_group controls.

        Had to rewrite due to  getting rid encode and decode functions.
        """
        conditioning = self.encode(features, training=training)
        processor_inputs = self.decoder(conditioning)
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
        #
        decoder = ddsp.training.decoders.RnnFcDecoder(
            rnn_channels = 256,
            rnn_type = 'gru',
            ch = 256,
            layers_per_stack = 1,
            output_splits = (('amps', 1),
                             ('harmonic_distribution', 45),
                             ('noise_magnitudes', 45)))
        return decoder

    def buildProcessorGroup(self, n_samples):
        """ Create actual synth structure (highly customizable)

        Must pass in n_samples (DDSP_DATASET.n_samples)

        """
        # Create Processors.

        additive = ddsp.synths.Additive(n_samples=n_samples,
                                        sample_rate=sample_rate, # THIS IS DEFINED GLOBALLY IN my_ddsp_utils
                                        name='additive')
        noise = ddsp.synths.FilteredNoise(window_size=0,
                                          initial_bias=-10.0,
                                          name='noise')
        add = ddsp.processors.Add(name='add')
        reverb = ddsp.effects.Reverb(name='reverb', trainable=True)

        # package them together in a  ProcessorGroup.
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
