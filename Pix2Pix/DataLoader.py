import os
import glob
import tensorflow as tf
import numpy as np
from Options import Options
import random


class Loader(Options):
    def __init__(self):
        super(Loader, self).__init__()

        folder_Path = f'{self.root}/A/live'

        A_folders = glob.glob(f'{folder_Path}/*')
        B_folders = glob.glob(f'{folder_Path}/*')

        B_folders = shuffle_folder(A_folders, B_folders)

        self.A = []
        self.B = []

        for folder in A_folders:
            temp = glob.glob(f'{folder}/*.png')
            self.A = self.A + temp

        for folder in B_folders:
            temp = glob.glob(f'{folder}/*.png')
            self.B = self.B + temp

        self.A_ds = tf.data.Dataset.from_tensor_slices(self.A)
        self.B_ds = tf.data.Dataset.from_tensor_slices(self.B)

    def decode_img(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, 3)
        img = tf.image.resize(img, [256, 256])
        img = (img / 127.5) - 1

        return img

    def load(self):
        A_ds = self.A_ds.map(self.decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        B_ds = self.B_ds.map(self.decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return tf.data.Dataset.zip((A_ds, B_ds))

    def __call__(self, *args, **kwargs):
        return self.load()


def configure_for_performance(ds, cnt, shuffle=False):
    if shuffle==True:
        ds = ds.shuffle(buffer_size=cnt)
        ds = ds.batch(1)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    elif shuffle==False:
        ds = ds.batch(1)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def shuffle_folder(A_folders, B_folders):
    random.shuffle(B_folders)
    for i in range(len(A_folders)):
        if A_folders[i] == B_folders[i]:
            return shuffle_folder(A_folders, B_folders)
    return B_folders
