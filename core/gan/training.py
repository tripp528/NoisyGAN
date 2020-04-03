from core.utils import *
### New Import
from scipy.io import wavfile
import pandas as pd

def train_discriminator(disc, opt, dataset_iter, iters=1):
    # unfreeze weights
    disc.trainable = True

    for i in range(iters): #TODO: shuffle batch around!
        batch = next(dataset_iter)
        # train_step
        grad_clip_norm = 3.0
        with tf.GradientTape() as tape:
            pred = disc(batch,training=True)
            total_loss = tf.reduce_sum(disc.losses)
        grads = tape.gradient(total_loss, disc.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        opt.apply_gradients(zip(grads, disc.trainable_variables))

        # logging.info("Disc Loss: " + str(total_loss.numpy()))
        # outfile.write(str(total_loss.numpy()) + '\n')


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
        grads = tape.gradient(total_loss, gan_model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        opt.apply_gradients(zip(grads, gan_model.trainable_variables))

        # logging.info("Gen Loss: " + str(total_loss.numpy()))
        # outfile.write(str(total_loss.numpy()) + '\n')

def train_gan(gan_model,
              opt,
              combined_iter,
              batch_size=8,
              total_iters=1,
              disc_iters=1,
              gen_iters=1,
              model_dir=None):
    # load losses and weights if they exist
    if model_dir:
        if not maybe_make_dir(model_dir):
            losses_df = pd.read_csv(model_dir + "losses.csv")
        else:
            losses_df = pd.DataFrame({"disc": [], "gen": []})

    # main loop
    for i in range(len(losses_df), len(losses_df) + total_iters):
        logging.info("----- GAN Step " + str(i) + " -----")
        train_discriminator(gan_model.disc, opt, combined_iter, iters=disc_iters)
        train_generator(gan_model,opt,iters=gen_iters,batch_size=batch_size)
        if model_dir:
            gan_checkpoint(model_dir, gan_model, i, losses_df)

def gan_checkpoint(model_dir,
                   gan_model,
                   i,
                   losses_df,
                   audio_period=1,
                   loss_period=1):
    # always save losses to dataframe
    losses_df = losses_df.append(
        {
            "disc" : str(tf.reduce_sum(gan_model.disc.losses).numpy()),
            "gen" : str(tf.reduce_sum(gan_model.losses).numpy())
        },
        ignore_index=True
    )

    # save audio
    if (i % audio_period) == 0:
        audio_path = model_dir + "chkpt-iter-" + str(i) + ".wav"
        wavfile.write(audio_path, DEFAULT_SAMPLE_RATE, gan_model.gen.generate()['audio'].numpy())

    # save lists in file every so often
    if (i % loss_period) == 0:
        losses_df.to_csv(model_dir + "losses.csv",index=False)
