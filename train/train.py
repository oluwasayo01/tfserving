import tensorflow as tf
import argparse
import pathlib

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32
    )
    parser.add_argument(
        '--img-height',
        type=float,
        default=180
    )
    parser.add_argument(
        '--img-width',
        type=float,
        default=180
    )


    args, _ = parser.parse_known_args()
    return args


def build_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(  )
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model

def split_data(directory=None, val_split=0.2, subset="training", img_height=180, img_width=180, batch_size=32):
    path_dir = pathlib.Path(directory)
    data = tf.keras.preprocessing.image_dataset_from_directory(
        path_dir,
        validation_split=val_split,
        subset=subset,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    if subset == "training":
        data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    elif subset == "validation":
        data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

if __name__ == '__main__':
    
    args = parse_arguments()
    trainset = split_data("flowers/", img_height=args.img_height, img_width=args.img_width, batch_size=args.batch_size)
    
    validationset = split_data("flowers/", subset="validation", img_height=args.img_height, img_width=args.img_width, batch_size=args.batch_size)

    model = build_cnn_model()

    model.fit(
        trainset,
        validation_data=validationset,
        epochs=10
    )
