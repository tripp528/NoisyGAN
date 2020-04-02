from core.utils import *
### New Import
from scipy.io import wavfile

def train_discriminator(disc, opt, dataset_iter, iters=1):
    # unfreeze weights
    disc.trainable = True

    for i in range(iters): #todo: shuffle batch around!
        batch = next(dataset_iter)
        # train_step
        grad_clip_norm = 3.0
        with tf.GradientTape() as tape:
            pred = disc(batch,training=True)
            # print(pred,batch['label'])
            total_loss = tf.reduce_sum(disc.losses)
            logging.info("Disc Loss: " + str(total_loss.numpy()))
        grads = tape.gradient(total_loss, disc.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        opt.apply_gradients(zip(grads, disc.trainable_variables))


def train_generator(gan_model, opt, batch_size=8, iters=1):
    # freeze weights
    gan_model.disc.trainable = False

    # discriminator weights are frozen - just being used as loss function
    for i in range(iters):
        # train_step
        grad_clip_norm = 3.0
        with tf.GradientTape() as tape:
            pred = gan_model(None,batch_size=batch_size)
            # print(pred)
            total_loss = tf.reduce_sum(gan_model.losses)
            logging.info("Gen Loss: " + str(total_loss.numpy()))
        grads = tape.gradient(total_loss, gan_model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        opt.apply_gradients(zip(grads, gan_model.trainable_variables))

def train_gan(gan_model, opt, combined_iter, batch_size=8, iters=1, disc_iters = 1, gen_iters = 1, checkpoints = None):
    for i in range(iters):
        logging.info("discriminator step" + str(i) + ":")
        train_discriminator(gan_model.disc, opt, combined_iter, iters=disc_iters)
        train_generator(gan_model,opt,iters=gen_iters,batch_size=batch_size)
        params = {"model":gan_model, "iter":i, 'disc_iters':disc_iters, 'gen_iters':gen_iters, 'loss':tf.reduce_sum(gan_model.losses).numpy()}
        if checkpoints:
            for checkpoint in checkpoints:
                checkpoint.call(model = gan_model,
                                    iter = i,
                                    disc_iters = disc_iters,
                                    gen_iters = gen_iters,
                                    loss = tf.reduce_sum(gan_model.losses).numpy())


# TODO: Make better
#       Work with other metrics (mark loss in filename (?))
class AudiofileCheckpoint():
    # Class to periodically save generator outputs
    def __init__(self, outpath, outname, sr, period=1):
        self.outpath = outpath
        self.outname = outname
        self.period = period
        self.sr = sr

    def call(self, **kwargs):
        if (kwargs['iter'] % self.period) == 0:
            out = (self.outpath+self.outname).format(kwargs)

            wavfile.write(out, self.sr, kwargs['model'].gen.generate()['audio'].numpy())
            print("Saved ", (out))

class LossCheckpoint():
    # Class to periodically save losses to file
    def __init__(self, disc_outpath, gen_outpath, sr, period=1):
        self.disc_outpath = disc_outpath
        self.gen_outpath = gen_outpath
        self.period = period
        self.sr = sr

        self.disc_loss = []
        self.gen_loss = []

    def call(self, **kwargs):
        if (kwargs['iter'] % self.period) == 0:
            self.disc_loss.append(str(tf.reduce_sum(kwargs['model'].disc.losses).numpy()))
            self.gen_loss.append(str(tf.reduce_sum(kwargs['model'].losses).numpy()))

    def savetofile(self):
        disc_outfile = open(self.disc_outpath, "a")
        gen_outfile = open(self.gen_outpath, "a")

        disc_outstring = ""
        gen_outstring = ""

        for loss in self.disc_loss:
            disc_outstring += str(loss)
            disc_outstring += '\n'

        for loss in self.gen_loss:
            gen_outstring += str(loss)
            gen_outstring += '\n'

        disc_outfile.write(disc_outstring)
        gen_outfile.write(gen_outstring)

        disc_outfile.close()
        gen_outfile.close()
