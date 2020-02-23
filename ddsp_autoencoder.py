from ddsp_dataset import *
import gin
from absl import logging

class DDSP_AUTOENCODER:
    def __init__(self, ddsp_dataset, model_dir, restore=False):
        self.ddsp_dataset = ddsp_dataset
        self.model_dir = model_dir
        # get distribution strategy (change if using gpus/tpus)
        self.strategy = ddsp.training.train_util.get_strategy()
        self.buildModel()

        # restore from checkpoint
        if restore == True:
            self.model_dir = find_model_dir(self.model_dir)
            ckpt = ddsp.training.train_util.get_latest_chekpoint(self.model_dir)
            print("restoring...",ckpt)
            self.model.restore(ckpt)

        # get trainer
        with self.strategy.scope():
            self.trainer = ddsp.training.train_util.Trainer(self.model, self.strategy, learning_rate=1e-3)

        # Build model, easiest to just run forward pass.
        dataset = self.trainer.distribute_dataset(self.ddsp_dataset.getDataset())
        self.trainer.build(next(iter(dataset)))

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

        # Create ProcessorGroup.
        dag = [(additive, ['amps', 'harmonic_distribution', 'f0_hz']),
               (noise, ['noise_magnitudes']),
               (add, ['noise/signal', 'additive/signal'])]

        processor_group = ddsp.processors.ProcessorGroup(dag=dag,
                                                         name='processor_group')


        # Loss_functions
        spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1',
                                                 mag_weight=1.0,
                                                 logmag_weight=1.0)

        with self.strategy.scope():
            # Put it together in a model.
            self.model = ddsp.training.models.Autoencoder(preprocessor=preprocessor,
                                             encoder=None,
                                             decoder=decoder,
                                             processor_group=processor_group,
                                             losses=[spectral_loss])
            # self.trainer = ddsp.training.train_util.Trainer(self.model, self.strategy, learning_rate=1e-3)

    def predict(self, sampleNum=0):
        # Run a batch of predictions.
        sample = self.ddsp_dataset.getSample(sampleNum=sampleNum)
        start_time = time.time()
        controls =  self.model.get_controls(sample)
        audio_gen = controls['processor_group']['signal']
        print('Prediction took %.1f seconds' % (time.time() - start_time))
        return sample["audio"], audio_gen

    def train(self, iterations=10):
        train_helper(self.ddsp_dataset.data_provider,
              self.trainer,
              batch_size=2,
              num_steps=iterations,
              steps_per_summary=5,
              steps_per_save=5,
              model_dir=self.model_dir)

def train_helper(data_provider,
                  trainer,
                  batch_size=32,
                  num_steps=1000000,
                  steps_per_summary=300,
                  steps_per_save=300,
                  model_dir='~/tmp/ddsp'):


  import tensorflow.compat.v2 as tfcv2

  """Main training loop."""
  # Get a distributed dataset.
  dataset = data_provider.get_batch(batch_size, shuffle=True, repeats=-1)
  dataset = trainer.distribute_dataset(dataset)
  dataset_iter = iter(dataset)

  # Build model, easiest to just run forward pass.
  trainer.build(next(dataset_iter))

  # Load latest checkpoint if one exists in model_dir.
  trainer.restore(model_dir)

  # Create training loss metrics.
  avg_losses = {name: tfcv2.keras.metrics.Mean(name=name, dtype=tf.float32)
                for name in trainer.model.loss_names}

  # Set up the summary writer and metrics.
  summary_dir = os.path.join(model_dir, 'summaries', 'train')
  summary_writer = tfcv2.summary.create_file_writer(summary_dir)

  # Save the gin config.
  ddsp.training.train_util.write_gin_config(summary_writer, model_dir, trainer.step.numpy())

  # Train.
  with summary_writer.as_default():
    for _ in range(num_steps):
      step = trainer.step

      # Take a step.
      losses = trainer.train_step(dataset_iter)

      # Update metrics.
      for k, v in losses.items():
        avg_losses[k].update_state(v)

      # Log the step.
      logging.info('Step:%d Loss:%.2f', step, losses['total_loss'])
      print('Step:%d Loss:%.2f' % (step, losses['total_loss']))

      # Write Summaries.
      if step % steps_per_summary == 0:
        for k, metric in avg_losses.items():
          tfcv2.summary.scalar('losses/{}'.format(k), metric.result(), step=step)
          metric.reset_states()

      # Save Model.
      if step % steps_per_save == 0:
        trainer.save(model_dir)
        summary_writer.flush()

  logging.info('Training Finished!')
  print('Training Finished!')
