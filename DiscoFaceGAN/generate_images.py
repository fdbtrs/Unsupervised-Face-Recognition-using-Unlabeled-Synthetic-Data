# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script for generating an image using pre-trained generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import tensorflow as tf
from os.path import join as ojoin
import argparse
from tqdm import tqdm


def open_file_or_url(file_or_url):
    if dnnlib.util.is_url(file_or_url):
        return dnnlib.util.open_url(file_or_url, cache_dir=config.cache_dir)
    return open(file_or_url, "rb")


def load_pkl(file_or_url):
    with open_file_or_url(file_or_url) as file:
        return pickle.load(file, encoding="latin1")


# define mapping network from z space to lambda space
def coeff_decoder(z, ch_depth=3, ch_dim=512, coeff_length=128):
    with tf.variable_scope("stage1"):
        with tf.variable_scope("decoder"):
            y = z
            for i in range(ch_depth):
                y = tf.layers.dense(y, ch_dim, tf.nn.relu, name="fc" + str(i))

            x_hat = tf.layers.dense(y, coeff_length, name="x_hat")
            x_hat = tf.stop_gradient(x_hat)

    return x_hat


# restore pre-trained weights
def restore_weights_and_initialize():
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()

    # add batch normalization params into trainable variables
    bn_moving_vars = [g for g in g_list if "moving_mean" in g.name]
    bn_moving_vars += [g for g in g_list if "moving_variance" in g.name]
    var_list += bn_moving_vars

    var_id_list = [v for v in var_list if "id" in v.name and "stage1" in v.name]
    var_exp_list = [v for v in var_list if "exp" in v.name and "stage1" in v.name]
    var_gamma_list = [v for v in var_list if "gamma" in v.name and "stage1" in v.name]
    var_rot_list = [v for v in var_list if "rot" in v.name and "stage1" in v.name]

    saver_id = tf.train.Saver(var_list=var_id_list, max_to_keep=100)
    saver_exp = tf.train.Saver(var_list=var_exp_list, max_to_keep=100)
    saver_gamma = tf.train.Saver(var_list=var_gamma_list, max_to_keep=100)
    saver_rot = tf.train.Saver(var_list=var_rot_list, max_to_keep=100)

    saver_id.restore(tf.get_default_session(), "./vae/weights/id/stage1_epoch_395.ckpt")
    saver_exp.restore(
        tf.get_default_session(), "./vae/weights/exp/stage1_epoch_395.ckpt"
    )
    saver_gamma.restore(
        tf.get_default_session(), "./vae/weights/gamma/stage1_epoch_395.ckpt"
    )
    saver_rot.restore(
        tf.get_default_session(), "./vae/weights/rot/stage1_epoch_395.ckpt"
    )


def z_to_lambda_mapping(latents):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        with tf.variable_scope("id"):
            id_coeff = coeff_decoder(
                z=latents[:, :128], coeff_length=160, ch_dim=512, ch_depth=3
            )
        with tf.variable_scope("exp"):
            exp_coeff = coeff_decoder(
                z=latents[:, 128 : 128 + 32], coeff_length=64, ch_dim=256, ch_depth=3
            )
        with tf.variable_scope("gamma"):
            gamma_coeff = coeff_decoder(
                z=latents[:, 128 + 32 : 128 + 32 + 16],
                coeff_length=27,
                ch_dim=128,
                ch_depth=3,
            )
        with tf.variable_scope("rot"):
            rot_coeff = coeff_decoder(
                z=latents[:, 128 + 32 + 16 : 128 + 32 + 16 + 3],
                coeff_length=3,
                ch_dim=32,
                ch_depth=3,
            )

        input_coeff = tf.concat([id_coeff, exp_coeff, rot_coeff, gamma_coeff], axis=1)
        return input_coeff


# generate images using attribute-preserving truncation trick
def truncate_generation(
    gen, inputcoeff, rate=0.7, dlatent_average_id=None, batchsize=32
):

    if dlatent_average_id is None:
        url_pretrained_model_ffhq_average_w_id = (
            "https://drive.google.com/uc?id=17L6-ENX3NbMsS3MSCshychZETLPtJnbS"
        )
        with dnnlib.util.open_url(
            url_pretrained_model_ffhq_average_w_id, cache_dir=config.cache_dir
        ) as f:
            dlatent_average_id = np.loadtxt(f)
    dlatent_average_id = np.reshape(dlatent_average_id, [1, 14, 512]).astype(np.float32)
    dlatent_average_id = tf.constant(dlatent_average_id)

    inputcoeff_id = tf.concat([inputcoeff[:, :160], tf.zeros([batchsize, 126])], axis=1)
    dlatent_out = gen.components.mapping.get_output_for(
        inputcoeff, None, is_training=False, is_validation=True
    )  # original w space output
    dlatent_out_id = gen.components.mapping.get_output_for(
        inputcoeff_id, None, is_training=False, is_validation=True
    )

    dlatent_out_trun = dlatent_out + (dlatent_average_id - dlatent_out_id) * (1 - rate)
    dlatent_out_final = tf.concat(
        [dlatent_out_trun[:, :8, :], dlatent_out[:, 8:, :]], axis=1
    )  # w space latent vector with truncation trick

    fake_images_out = gen.components.synthesis.get_output_for(
        dlatent_out_final, randomize_noise=False
    )
    fake_images_out = tf.clip_by_value((fake_images_out + 1) * 127.5, 0, 255)
    fake_images_out = tf.transpose(fake_images_out, perm=[0, 2, 3, 1])

    return fake_images_out


# calculate average w space latent vector with zero expression, lighting, and pose.
def get_model_and_average_w_id(model_name):
    _, _, gen = load_pkl(model_name)
    average_w_name = model_name.replace(".pkl", "-average_w_id.txt")
    if not os.path.isfile(average_w_name):
        print("Calculating average w id...\n")
        latents = tf.placeholder(
            tf.float32, name="latents", shape=[1, 128 + 32 + 16 + 3]
        )
        noise = tf.placeholder(tf.float32, name="noise", shape=[1, 32])
        input_coeff = z_to_lambda_mapping(latents)
        input_coeff_id = input_coeff[:, :160]
        input_coeff_w_noise = tf.concat(
            [input_coeff_id, tf.zeros([1, 64 + 27 + 3]), noise], axis=1
        )
        dlatent_out = gen.components.mapping.get_output_for(
            input_coeff_w_noise, None, is_training=False, is_validation=True
        )
        restore_weights_and_initialize()
        np.random.seed(1)
        average_w_id = []
        for _ in range(50000):
            lats = np.random.normal(size=[1, 128 + 32 + 16 + 3])
            noise_ = np.random.normal(size=[1, 32])
            w_out = tflib.run(dlatent_out, {latents: lats, noise: noise_})
            average_w_id.append(w_out)

        average_w_id = np.concatenate(average_w_id, axis=0)
        average_w_id = np.mean(average_w_id, axis=0)
        np.savetxt(average_w_name, average_w_id)
    else:
        average_w_id = np.loadtxt(average_w_name)

    return gen, average_w_id


def parse_args():
    desc = "Disentangled face image generation"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--factor",
        type=int,
        default=0,
        help="factor variation mode. 0 = all, 1 = expression, 2 = lighting, 3 = pose.",
    )
    parser.add_argument(
        "--subject", type=int, default=100000, help="how many subjects to generate."
    )
    parser.add_argument("--offset", type=int, default=0, help="naming offset")
    parser.add_argument(
        "--model",
        type=str,
        default="pretrained/network-snapshot-020126.pkl",
        help="pkl file name of the generator. If None, use the default pre-trained model.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/data/maklemt/synthetic_imgs/DiscoFaceGAN_large",
        help="Where to save the images",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=128,
        help="batch size and number of augmentations per subject",
    )

    return parser.parse_args()


def load_gen(url):
    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, gen = pickle.load(f)
    return gen


def main():
    args = parse_args()
    if args is None:
        exit()
    # save path for generated images
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    bs = args.batchsize
    offset = args.offset

    tflib.init_tf()

    with tf.device("/gpu:0"):

        # Use default pre-trained model
        if args.model is None:
            url_pretrained_model_ffhq = (
                "https://drive.google.com/uc?id=1nT_cf610q5mxD_jACvV43w4SYBxsPUBq"
            )
            gen = load_gen(url_pretrained_model_ffhq)
            average_w_id = None

        else:
            gen, average_w_id = get_model_and_average_w_id(args.model)
        # gen = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
        # average_w_id = average w space latent vector with zero expression, lighting, and pose.

        # Print network details.
        gen.print_layers()

        # Pick latent vector.
        latents = tf.placeholder(
            tf.float32, name="latents", shape=[bs, 128 + 32 + 16 + 3]
        )
        noise = tf.placeholder(tf.float32, name="noise", shape=[bs, 32])
        input_coeff = z_to_lambda_mapping(latents)
        input_coeff_w_noise = tf.concat([input_coeff, noise], axis=1)

        # Generate images
        fake_images_out = truncate_generation(
            gen, input_coeff_w_noise, dlatent_average_id=average_w_id, batchsize=bs
        )

    restore_weights_and_initialize()

    print("Start generating images...")
    np.random.seed(42)
    for i in tqdm(range(args.subject)):
        person = "%07d" % (i + offset)
        save_dir = ojoin(save_path, person)
        os.makedirs(save_dir, exist_ok=True)
        lats1 = np.random.normal(size=[1, 128 + 32 + 16 + 3])
        lats1 = np.repeat(lats1, bs, axis=0)
        noise_ = np.random.normal(size=[1, 32])
        noise_ = np.repeat(noise_, bs, axis=0)

        lats2 = np.random.normal(size=[bs, 32 + 16 + 3])
        if args.factor == 0:  # change all factors
            lats = np.concatenate([lats1[:, :128], lats2], axis=1)
        elif args.factor == 1:  # change expression only
            lats = np.concatenate(
                [lats1[:, :128], lats2[:, :32], lats1[:, 128 + 32 :]], axis=1
            )
        elif args.factor == 2:  # change lighting only
            lats = np.concatenate(
                [
                    lats1[:, : 128 + 32],
                    lats2[:, 32 : 32 + 16],
                    lats1[:, 128 + 32 + 16 :],
                ],
                axis=1,
            )
        elif args.factor == 3:  # change pose only
            lats = np.concatenate(
                [lats1[:, : 128 + 32 + 16], lats2[:, 32 + 16 : 32 + 16 + 3]], axis=1
            )
        fake = tflib.run(fake_images_out, {latents: lats, noise: noise_})
        j = 0
        for fa in fake:
            PIL.Image.fromarray(fa.astype(np.uint8), "RGB").save(
                ojoin(save_dir, "%07d_%03d.jpg" % (i + offset, j))
            )
            j += 1
    print("all images created")


if __name__ == "__main__":
    main()
