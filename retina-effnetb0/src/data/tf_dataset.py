# make sure every file in /src uses tf_keras OR keras consistently. We chose tf_keras.
import tensorflow as tf
import pandas as pd

AUTOTUNE = tf.data.AUTOTUNE

class DataBuilder:
    def __init__(self, input_shape=(380, 380, 3), augment=True):
        self.h, self.w, self.c = input_shape
        self.augment = augment
        self._build_aug()

    def _build_aug(self):
        if self.augment:
            self.aug = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
            ])
        else:
            self.aug = None

    def parse(self, root_dir, row):
        img_path = tf.strings.join([root_dir, "/", "images", "/", row["Image name"]])
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        # img = tf.image.convert_image_dtype(img, tf.float32)
        # replace with (leave in 0..255 for EfficientNet’s built-in rescale) ✅
        img = tf.cast(img, tf.float32)   # values ~0..255 after decode/resize
        img = tf.image.resize(img, (self.h, self.w), antialias=True)
        if self.aug is not None:
            img = self.aug(img, training=True)

        y_grade = tf.cast(row["Retinopathy grade"], tf.int32)
        y_edema = tf.cast(row["Risk of macular edema"], tf.float32)

        # <<< IMPORTANT: return a tuple, NOT a dict
        return img, (y_grade, y_edema)

    def dataframe_to_ds(self, csv_path, root_dir, shuffle, batch):
        df = pd.read_csv(csv_path)
        ds = tf.data.Dataset.from_tensor_slices({
            "Image name": df["Image name"].values,
            "Retinopathy grade": df["Retinopathy grade"].values,
            "Risk of macular edema": df["Risk of macular edema"].values,
        })
        ds = ds.map(lambda r: self.parse(root_dir, r), num_parallel_calls=AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(2048, reshuffle_each_iteration=True)
        ds = ds.batch(batch).prefetch(AUTOTUNE)
        return ds
