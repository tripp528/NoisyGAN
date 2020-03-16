from .solo_autoencoder import *
from .gan import *

class DDSP_TRAINER(ddsp.training.train_util.Trainer):
    """ Extension of Trainer, which is defined as:
        Class to bind an optimizer, model, strategy, and training step function.

    This class aims to give default arguments to the trainer.
        The following functionality is added:

    1) Easy to use train and predict functions that take in a DDSP_DATASET object.

    2) This class bundles in a model directory (model_dir) as well. In ddsp.train_util,
        this is not passed in until you actually call the train function (outside of
        Trainer class). This enables automatic saving, and automatic resotoring
        of your model from any checkpoints already in the model directory.

    3) Takes gpu arguments directly, and constructs a strategy internally from that.
        This eliminates the need to construct a strategy in your notebook / script.

    4) Constructs a model internally, without outside definition. This means you don't
        need to use the model class directly.
        TODO: take in a string, and construct a different model depending on the string.
    """
    def __init__(self, model_dir, restore=False, gpus=None, model_type="solo"):
        self.model_dir, self.found_model_dir = find_model_dir(model_dir)
        logging.info("MODEL_DIR: " + str(self.model_dir) + " FOUND: " + str(self.found_model_dir))
        self.strategy = ddsp.training.train_util.get_strategy(gpus=gpus) # get distribution strategy (change if using gpus/tpus)
        self.model = self.buildModel(model_type)

        super().__init__(self.model,
                        self.strategy,
                        checkpoints_to_keep=10,
                        learning_rate=0.001,
                        lr_decay_steps=10000,
                        lr_decay_rate=0.98,
                        grad_clip_norm=3.0)

        # AUTOMATICALLY restore from checkpoint. If you want to not restore, clear the
        # model dir or set a new model dir.

        # Build model, easiest to just run forward pass.
        # trainer.build(next(dataset_iter))  TODO: does this fix model_dir problem?

        if (self.found_model_dir):
            self.auto_restore()

    def buildModel(self,model_type):
        with self.strategy.scope():
            if model_type ==  "solo":
                return Solo_Autoencoder()
            elif model_type == "gan":
                return GAN()

    def predict(self, dataset, sampleNum=0):
        """Run a batch of predictions."""
        sample = dataset.getSample(sampleNum=sampleNum)
        start_time = time.time()
        controls =  self.model.get_controls(sample)
        audio_gen = controls['processor_group']['signal']
        logging.info('Prediction took %.1f seconds' % (time.time() - start_time))
        return sample["audio"], audio_gen

    def train(self, dataset, iterations=10000):
        """
        Calls ddsp.training.train_util.train, and passes in self as the trainer

            default stuff:
                          batch_size=32,
                          num_steps=1000000,
                          steps_per_summary=300,
                          steps_per_save=300,
                          model_dir='~/tmp/ddsp'
        """
        # self.auto_restore()
        ddsp.training.train_util.train(
                dataset.data_provider,
                self,
                batch_size=32,
                num_steps=iterations,
                steps_per_summary=10,
                steps_per_save=1,
                model_dir=self.model_dir)

    def auto_restore(self):
        ckpt = ddsp.training.train_util.get_latest_chekpoint(self.model_dir)
        if ckpt:
            logging.info("restoring... "+ckpt)
            self.restore(ckpt)
        else:
            logging.info("no checkpoint found in model_dir")
