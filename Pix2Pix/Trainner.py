import tensorflow as tf
import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from Model import Model
import re
from DataLoader import Loader, configure_for_performance
from Options import Options
# from IPython import display


class Trainner(Options):
    def __init__(self):
        super(Trainner, self).__init__()
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.builder = Model()

        self.generator = self.builder.Generator()
        self.discriminator = self.builder.Discriminator()

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1)

        log_dir = ".log/fit/1"

        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def mkdir(self):
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)

        if not os.path.exists(self.OUTPUT_DIR_CKP):
            os.makedirs(self.OUTPUT_DIR_CKP)

        if not os.path.exists(self.OUTPUT_DIR_SAMPLE):
            os.makedirs(self.OUTPUT_DIR_SAMPLE)

        if not os.path.exists(self.OUTPUT_DIR_LOSS):
            os.makedirs(self.OUTPUT_DIR_LOSS)

        if not os.path.exists(self.OUTPUT_DIR_TEST):
            os.makedirs(self.OUTPUT_DIR_TEST)

    def save_sampling(self, img, ep, step, names):
        # output = self.generator(img)

        # output = output * 255.0
        output = tf.clip_by_value(img * 255.0, 0, 255).numpy()
        img_Name = re.split(r'[\\]', names)[-1]
        img_Name = re.compile('.png').sub('', img_Name)
        for i in range(len(output)):
            cv2.imwrite(f'{self.OUTPUT_DIR_SAMPLE}/{ep}_{step}_{img_Name}.png', cv2.cvtColor(output[i], cv2.COLOR_RGB2BGR))

    def save_test(self, img, names):
        output = tf.clip_by_value(img * 255.0, 0, 255).numpy()
        img_Name = re.split(r'[\\]', names)[-1]

        for i in range(len(img)):
            cv2.imwrite(f'{self.OUTPUT_DIR_TEST}/{img_Name}', cv2.cvtColor(output[i], cv2.COLOR_RGB2BGR))

    def save_loss(self, ep, disc_loss, gen_loss):
        print(f'||Epoch {ep} / {self.epochs}|| '
              f'\n || GenLoss {np.mean(gen_loss)}|| '
              f'\n || Disc Loss {np.mean(disc_loss)}')

        fig, ax = plt.subplots()
        plt.plot(disc_loss, label='Discriminator', alpha=0.6)
        plt.plot(gen_loss, label='Generator', alpha=0.6)
        plt.title('Losses')
        plt.legend()
        plt.savefig(f'{self.OUTPUT_DIR_LOSS}/losses_{str(ep)}.png')
        # plt.show()
        # plt.close()

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    @tf.function
    def train_step(self, img, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(img, training=True)

            disc_real_output = self.discriminator([img, target], training=True)
            disc_generated_output = self.discriminator([img, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)

        return gen_output, gen_total_loss, disc_loss

    def test_step(self, model, img):
        gen_output = model(img, training=False)

        return gen_output

    def training(self):
        self.mkdir()
        ds_loder = Loader()
        checkpoint_prefix = os.path.join(self.OUTPUT_DIR_CKP, "ckpt")
        checkpint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)
        gen_losses = []
        disc_losses = []

        for ep in range(self.epochs):
            print(f'||  Epoch : {ep}/{self.epochs}  ||')
            full_ds = ds_loder()

            full_ds = configure_for_performance(full_ds, self.A_cnt, shuffle=False)
            full_ds_iter = iter(full_ds)

            start = time.time()
            # display.clear_output(wait=True)

            for step in range(self.A_cnt//self.BatchSZ):
                img, target = next(full_ds_iter)
                gen_output, gen_total_loss, disc_loss = self.train_step(img, target, ep)

                gen_losses.append(gen_total_loss)
                disc_losses.append(disc_loss)

                if step % 100 == 0:
                    print(f'||  step : {step}/{self.A_cnt//self.BatchSZ}    |   GenLoss : {gen_total_loss}  |   DiscLoss : {disc_loss}||')
                    self.save_sampling(gen_output, ep, step, ds_loder.A[step])

            self.save_loss(ep, gen_losses, disc_losses)

            checkpint.save(file_prefix=checkpoint_prefix)

    def testing(self):
        builder = Model()
        generator = builder.Generator()
        discriminator = builder.Discriminator()

        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1)

        checkpoint_dir = f'E:/backup/PIX2PIX/ND/1-fold/2021-12-31_140013/checkpoints/ckpt-150'
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                        discriminator_optimizer=discriminator_optimizer,
                                        generator=generator,
                                        discriminator=discriminator)

        checkpoint.restore(checkpoint_dir)

        ds_loader = Loader()
        full_ds = ds_loader()

        full_ds = configure_for_performance(full_ds, self.A_cnt, shuffle=False)
        full_ds_iter = iter(full_ds)
        print('start testing!')

        for step in range(self.A_cnt//self.BatchSZ):
            img, target = next(full_ds_iter)
            output = self.test_step(generator, img)
            self.save_test(output, ds_loader.A[step])

            if step % 100 == 0:
                print(f'now step {step}/{self.BatchSZ}')
