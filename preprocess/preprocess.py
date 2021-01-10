import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pathlib
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    )

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':

    args = parse_arguments()
    dataset = tf.keras.utils.get_file(
        fname='flowers', origin=args.dataset, untar=True)
    data_dir = pathlib.Path(dataset)
    print(os.listdir())
