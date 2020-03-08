from ddsp_dataset import *
import gin

class DDSP_AUTOENCODER2(ddsp.training.train_util.Trainer):
    """Extension of Trainer - which combines model, (gpu) strategy, and optimizer"""

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
        TIME_STEPS = 1000

        # Create Neural Networks.
        preprocessor = ddsp.training.preprocessing.DefaultPreprocessor(time_steps=TIME_STEPS)

        decoder = ddsp.training.decoders.RnnFcDecoder(rnn_channels = 256,
                                        rnn_type = 'gru',
                                        ch = 256,
                                        layers_per_stack = 1,
                                        output_splits = (('amps', 1),
                                                         ('harmonic_distribution', 45),
                                                         ('noise_magnitudes', 45)))

        # Create Processors.
        additive = ddsp.synths.Additive(n_samples=self.ddsp_dataset.n_samples,
                                        sample_rate=sample_rate,
                                        name='additive')

        noise = ddsp.synths.FilteredNoise(window_size=0,
                                          initial_bias=-10.0,
                                          name='noise')
        add = ddsp.processors.Add(name='add')

        reverb = ddsp.effects.Reverb(name='reverb', trainable=True)

        # Create ProcessorGroup.
        dag = [(additive, ['amps', 'harmonic_distribution', 'f0_hz']),
               (noise, ['noise_magnitudes']),
               (add, ['noise/signal', 'additive/signal']),
               (reverb, ['add/signal'])]

        processor_group = ddsp.processors.ProcessorGroup(dag=dag,
                                                         name='processor_group')


        # Loss_functions
        spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1',
                                                 mag_weight=1.0,
                                                 logmag_weight=1.0)

        with self.strategy.scope():
            # Put it together in a model.
            self.model = Autoencoder2(preprocessor=preprocessor,
                                             encoder=None,
                                             decoder=decoder,
                                             processor_group=processor_group,
                                             losses=[spectral_loss])

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


@gin.configurable
class Autoencoder2(ddsp.training.models.Model):
  """Wrap the model function for dependency injection with gin."""

  def __init__(self,
               preprocessor=None,
               encoder=None,
               decoder=None,
               processor_group=None,
               losses=None,
               name='autoencoder2'):
    super().__init__(name=name, losses=losses)
    self.preprocessor = preprocessor
    self.encoder = encoder
    self.decoder = decoder
    self.processor_group = processor_group

  def encode(self, features, training=True):
    """Get conditioning by preprocessing then encoding."""
    conditioning = self.preprocessor(features, training=training)
    return conditioning if self.encoder is None else self.encoder(conditioning)

  def decode(self, conditioning, training=True):
    """Get generated audio by decoding than processing."""
    processor_inputs = self.decoder(conditioning, training=training)
    return self.processor_group(processor_inputs)

  def call(self, features, training=True):
    """Run the core of the network, get predictions and loss."""
    conditioning = self.encode(features, training=training)
    audio_gen = self.decode(conditioning, training=training)
    if training:
      self.add_losses(features['audio'], audio_gen)
    return audio_gen

  def get_controls(self, features, keys=None, training=False):
    """Returns specific processor_group controls."""
    conditioning = self.encode(features, training=training)
    processor_inputs = self.decoder(conditioning)
    controls = self.processor_group.get_controls(processor_inputs)
    return controls if keys is None else {k: controls[k] for k in keys}
