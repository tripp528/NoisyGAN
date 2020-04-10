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

        logging.info("Disc Loss: " + str(total_loss.numpy()))


def train_generator(gan_model, opt, iters=1):
    # freeze weights
    gan_model.disc.trainable = False

    # discriminator weights are frozen - just being used as loss function
    for i in range(iters):
        # train_step
        grad_clip_norm = 3.0
        with tf.GradientTape() as tape:
            pred = gan_model(None)
            # print(pred)
            total_loss = tf.reduce_sum(gan_model.losses)
        grads = tape.gradient(total_loss, gan_model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        opt.apply_gradients(zip(grads, gan_model.trainable_variables))

        logging.info("Gen Loss: " + str(total_loss.numpy()))

def train_gan(gan_model, gen_opt, disc_opt, combined_iter, **kwargs):
    DEFAULT_ARGS = {
        "model_dir": None,
        "total_iters": 1,
        "gen_iters": 1,
        "disc_iters": 1,
        "loss_period": 2,
        "audio_period": 2,
        "weights_period": 2,
    }
    kwargs = merge(DEFAULT_ARGS, kwargs)
    model_dir = kwargs["model_dir"]

    # load up metrics and weights, or initialize model_dir tree
    if model_dir:
        maybe_make_dir(model_dir)

        # maybe load losses
        if not os.path.exists(model_dir + "losses.csv"):
            losses_df = pd.DataFrame({"disc": [], "gen": []})
            losses_df.to_csv(model_dir + "losses.csv",index=False)
        else:
            losses_df = pd.read_csv(model_dir + "losses.csv")

        # maybe load weights
        maybe_make_dir(model_dir + "weights/")
        if os.listdir(model_dir + "weights/"):
            latest_checkpoint = tf.train.latest_checkpoint(model_dir + "weights/")
            gan_model.load_weights(latest_checkpoint)

        # make dir for audio samples
        maybe_make_dir(model_dir + 'samples/')

    else:
        # model_dir not specified
        losses_df = pd.DataFrame({"disc": [], "gen": []})

    # main loop
    for i in range(len(losses_df), len(losses_df) + kwargs["total_iters"]):
        logging.info("----- GAN Step " + str(i) + " -----")
        train_discriminator(gan_model.disc, disc_opt, combined_iter, iters=kwargs["disc_iters"])
        train_generator(gan_model,gen_opt,iters=kwargs["gen_iters"])
        if model_dir:
            gan_checkpoint(model_dir, gan_model, i, losses_df, kwargs)

def gan_checkpoint(model_dir, gan_model, i, losses_df, kwargs):

    # always append losses to dataframe
    disc_loss = str(tf.reduce_sum(gan_model.disc.losses).numpy())
    gen_loss = str(tf.reduce_sum(gan_model.losses).numpy())
    # logging.info("Disc loss: " + disc_loss + "Gen loss: " + gen_loss)
    losses_df = losses_df.append({"disc":disc_loss, "gen":gen_loss}, ignore_index=True)

    # save audio
    if (i % kwargs["audio_period"]) == 0:
        sample = gan_model.gen.generate()['audio'].numpy()
        play(sample)
        audio_path = model_dir + "samples/" + "chkpt-iter-" + str(i) + ".wav"
        wavfile.write(audio_path, DEFAULT_SAMPLE_RATE, sample)

    # export dataframe to csv every so often
    if (i % kwargs["loss_period"]) == 0:
        losses_df.to_csv(model_dir + "losses.csv",index=False)

    # save weights
    if (i % kwargs["weights_period"]) == 0:
        gan_model.save_weights(model_dir + "weights/iter" + str(i) +".ckpt")
