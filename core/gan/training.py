from core.utils import *

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

def train_gan(gan_model, opt, combined_iter, batch_size=8, iters=2):
    for i in range(iters):
        logging.info("discriminator step" + str(i) + ":")
        train_discriminator(gan_model.disc, opt, combined_iter, iters=1)
        train_generator(gan_model,opt,iters=1,batch_size=batch_size)
