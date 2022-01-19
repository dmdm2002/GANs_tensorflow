import os
import glob
import tensorflow as tf
import random
import numpy as np


class Loader(object):
    def __init__(self, root, shuffle=True):
        self.root = root
        A_folders = glob.glob(f'{root}/live/*')
        B_folders = glob.glob(f'{root}/live/*')

        def shuffle_list(A, B):
            random.shuffle(A)
            for i in range(len(A)):
                if A[i] == B[i]:
                    shuffle_list(A, B)
            return A

        A_imgs = []
        B_imgs = []
        for i in range(len(A_folders)):
            A = glob.glob(f'{A_folders[i]}/*.png')
            B = glob.glob(f'{B_folders[i]}/*.png')

            if shuffle:
                A = shuffle_list(A, B)
            A_imgs = A_imgs + A
            B_imgs = B_imgs + B
            
        self.A_imgs = np.array(A_imgs)

#         self.A_ds = tf.data.Dataset.from_tensor_slices(A_imgs)
#         self.B_ds = tf.data.Dataset.from_tensor_slices(B_imgs)


    def decode_img(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, 3)
        img = tf.image.resize(img, [28, 28]) / 255.

        return img

    def load(self):
        A_ds = tf.data.Dataset.from_tensor_slices(self.A_imgs)
        A_ds = A_ds.map(self.decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#         B_ds = self.B_ds.map(self.decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = tf.data.Dataset.zip((A_ds))

        return ds


def configure_for_performance(ds, cnt, batchz, shuffle=False):
    if shuffle==True:
        ds = ds.shuffle(buffer_size=cnt)
        ds = ds.batch(batchz)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    elif shuffle==False:
        ds = ds.batch(batchz)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds