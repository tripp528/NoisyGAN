from core.utils import *
### New Import
from scipy.io import wavfile
import pandas as pd

def train_disc(gan_model, opt, dataset_iter, iters=1, grad_clip_norm=3.0, add_noise=True):
    # unfreeze weights
    gan_model.disc.trainable = True

    for i in range(iters): #TODO: shuffle batch around!
        batch = dataset_iter.getNext()
        # Add noise to labels
        if add_noise:
            shape = batch['label'].numpy().shape
            noise_vec = (np.random.random_sample(shape) - 0.5) * 0.05
            batch['label'] = batch['label'] + noise_vec

        # train_step
        with tf.GradientTape() as tape:
            pred = gan_model.disc(batch,add_losses=True)
            logging.debug("disc_losses: " + str(gan_model.disc.losses))
            total_loss = tf.reduce_sum(gan_model.disc.losses)
        grads = tape.gradient(total_loss, gan_model.disc.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        opt.apply_gradients(zip(grads, gan_model.disc.trainable_variables))

    # logging.info("Disc Loss: " + str(total_loss.numpy()))
    return str(total_loss.numpy())


def train_gen(gan_model, opt, iters=1, grad_clip_norm=3.0):
    # freeze weights
    gan_model.disc.trainable = False

    # discriminator weights are frozen - just being used as loss function
    for i in range(iters):
        # train_step
        with tf.GradientTape() as tape:
            pred = gan_model(None)
            logging.debug("gen_losses: " + str(gan_model.losses))
            total_loss = tf.reduce_sum(gan_model.losses)

        grads = tape.gradient(total_loss, gan_model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        opt.apply_gradients(zip(grads, gan_model.trainable_variables))

    # logging.info("Gen Loss: " + str(total_loss.numpy()))
    return str(total_loss.numpy())

def train_gan(gan_model, combined_iter, **kwargs):
    DEFAULT_ARGS = {
        "model_dir": None,
        "total_iters": 10,
        "play_checkpints":False,
        # gen
        "gen_iters": 1,
        "gen_grad_clip_norm": 3.0,
        "gen_opt": tf.keras.optimizers.Adam(lr=0.001),
        #disc
        "disc_iters": 1,
        "disc_grad_clip_norm": 3.0,
        "disc_opt": tf.keras.optimizers.Adam(lr=0.001),
        "noisy_labels": True,
        # checkpoints
        "loss_period": 2,
        "audio_period": 2,
        "save_audio":True,
        "weights_period": 2,
        "auto_load_weights":False,
        "from_benchmark":True
    }
    kwargs = merge(DEFAULT_ARGS, kwargs)
    losses_df = initialize_model_dir(gan_model,kwargs)

    # main loop
    for i in range(len(losses_df), len(losses_df) + kwargs["total_iters"]):
        logging.info("----- GAN Step " + str(i) + " -----")
        disc_loss = train_disc(gan_model,
                               kwargs["disc_opt"],
                               combined_iter,
                               iters=kwargs["disc_iters"],
                               grad_clip_norm=kwargs["disc_grad_clip_norm"],
                               add_noise=kwargs["noisy_labels"])
        gen_loss = train_gen(gan_model,
                             kwargs["gen_opt"],
                             iters=kwargs["gen_iters"],
                             grad_clip_norm=kwargs["gen_grad_clip_norm"])
        logging.info("Disc loss: " + disc_loss + " Gen loss: " + gen_loss)
        if kwargs["model_dir"]:
            gan_checkpoint(gan_model, i, losses_df, gen_loss, disc_loss, kwargs)

def gan_checkpoint(gan_model, i, losses_df, gen_loss, disc_loss, kwargs):
    model_dir = kwargs["model_dir"]

    # always append losses to dataframe
    losses_df.loc[i] = {"disc":disc_loss, "gen":gen_loss}

    # save audio
    if (i % kwargs["audio_period"]) == 0 and i != 0:
        if kwargs["from_benchmark"]:
            sample = gan_model.gen.gen_from_benchmark()['audio'].numpy()
        else:
            sample = gan_model.gen.generate()['audio'].numpy()
        if kwargs['play_checkpoints']: play(sample)
        if kwargs["save_audio"]:
            audio_path = model_dir + "samples/" + "chkpt-iter-" + str(i) + ".wav"
            wavfile.write(audio_path, DEFAULT_SAMPLE_RATE, sample)
            logging.info("Sample saved to " + audio_path)

    # export dataframe to csv every so often
    if (i % kwargs["loss_period"]) == 0 and i != 0:
        logging.debug("Saving loss history to file..")
        losses_df.to_csv(model_dir + "losses.csv",index=False)

    # save weights
    if (i % kwargs["weights_period"]) == 0 and i != 0:
        logging.debug("Saving weights to checkpoint..")
        gan_model.save_weights(model_dir + "weights/iter" + str(i) +".ckpt")

def initialize_model_dir(gan_model, kwargs):
    """ Load up metrics and weights, or initialize model_dir tree """

    model_dir = kwargs["model_dir"]
    if model_dir:
        if kwargs["loss_period"] != kwargs["weights_period"]:
            logging.warning("If loss_period != weights_period, unexpected results may occur. "+\
                            "You should set auto_load_weights to False.")
        maybe_make_dir(model_dir)

        # maybe load losses
        if not os.path.exists(model_dir + "losses.csv"):
            losses_df = pd.DataFrame({"disc": [], "gen": []})
            losses_df.to_csv(model_dir + "losses.csv",index=False)
        else:
            losses_df = pd.read_csv(model_dir + "losses.csv")

        # maybe load weights
        if kwargs["auto_load_weights"]:
            maybe_make_dir(model_dir + "weights/")
            if os.listdir(model_dir + "weights/"):
                latest_checkpoint = tf.train.latest_checkpoint(model_dir + "weights/")
                logging.info("Loading weights from " + str(latest_checkpoint))
                gan_model.load_weights(latest_checkpoint)

        # make dir for audio samples
        maybe_make_dir(model_dir + 'samples/')

    else:
        # model_dir not specified
        losses_df = pd.DataFrame({"disc": [], "gen": []})

    return losses_df
