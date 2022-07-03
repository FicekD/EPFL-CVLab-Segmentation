import tensorflow as tf
from random import Random

from functools import partial

from .dataset_utils import load_tiff_images


def load_data(x_path, y_path):
    x = load_tiff_images(x_path)
    y = load_tiff_images(y_path)
    return x, y


def split(x, y, split_ratio):
    val_mask = int(x.shape[0] * split_ratio) * [True] + (x.shape[0] - int(x.shape[0] * split_ratio)) * [False]
    Random(41).shuffle(val_mask)
    train_mask = [not x for x in val_mask]
    x_train, y_train, x_val, y_val = x[train_mask], y[train_mask], x[val_mask], y[val_mask]
    return x_train, y_train, x_val, y_val


def transform_images(x):
    x = tf.cast(x, tf.float32) / 255.
    x = x[..., tf.newaxis]
    return x


def random_2d_crop(x, y, input_size=256, output_size=164):
    start_index_row = tf.random.uniform([], 0, x.shape[0] - input_size, dtype=tf.int32)
    start_index_col = tf.random.uniform([], 0, x.shape[1] - input_size, dtype=tf.int32)
    x = x[start_index_row:start_index_row+input_size, start_index_col:start_index_col+input_size, :]
    label_margin = (input_size - output_size) // 2
    y = y[start_index_row+label_margin:start_index_row+input_size-label_margin,
          start_index_col+label_margin:start_index_col+input_size-label_margin, :]
    return x, y


def image_to_patches(x, y, input_size=256, output_size=164):
    input_padding = (input_size - output_size) // 2
    bot_output_padding = output_size - (x.shape[0] % output_size) + input_padding
    right_output_padding = output_size - (x.shape[1] % output_size) + input_padding

    paddings = tf.constant([[input_padding, bot_output_padding],
                            [input_padding, right_output_padding]])
    padded_x = tf.pad(x[..., 0], paddings, 'SYMMETRIC')[..., tf.newaxis]
    padded_y = tf.pad(y[..., 0], paddings, 'SYMMETRIC')[..., tf.newaxis]

    patches_x, patches_y = list(), list()
    row = input_padding
    for _ in range((padded_x.shape[0] - 2 * input_padding) // output_size):
        col = input_padding
        for _ in range((padded_x.shape[1] - 2 * input_padding) // output_size):
            patch_x = padded_x[row-input_padding:row+output_size+input_padding, col-input_padding:col+output_size+input_padding, ...]
            patch_y = padded_y[row:row+output_size, col:col+output_size, ...]
            patches_x.append(patch_x)
            patches_y.append(patch_y)
            col += output_size
        row += output_size
    patches_x = tf.convert_to_tensor(patches_x, dtype=tf.float32)
    patches_y = tf.convert_to_tensor(patches_y, dtype=tf.float32)
    return patches_x, patches_y


def transform_batched_patches(x, y):
    x = tf.reshape(x, (-1, x.shape[-3], x.shape[-2], 1))
    y = tf.reshape(y, (-1, y.shape[-3], y.shape[-2], 1))
    return x, y


def construct_dataset(x, y, input_size=256, output_size=164, batch_size=16, testing=False, test_on_patches=True, shuffle=False, prefetch=False):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(lambda x, y: (transform_images(x), transform_images(y)))
    if testing and test_on_patches:
        dataset = dataset.map(partial(image_to_patches, input_size=input_size, output_size=output_size))
    else:
        dataset = dataset.map(partial(random_2d_crop, input_size=input_size, output_size=output_size))
    if shuffle:
        dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size)
    if testing and test_on_patches:
        dataset = dataset.map(transform_batched_patches)
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def init_2d_dataset(x_path, y_path, input_size=256, output_size=164, batch_size=16, validation_split_ratio=None, testing=False, validate_on_patches=True, shuffle=False, prefetch=False):
    x, y = load_data(x_path, y_path)
    if validation_split_ratio is None:
        dataset = construct_dataset(x, y, input_size=input_size, output_size=output_size, batch_size=batch_size, testing=testing, shuffle=shuffle, prefetch=prefetch)
        return dataset
    x_train, y_train, x_val, y_val = split(x, y, validation_split_ratio)
    train_dataset = construct_dataset(x_train, y_train, input_size=input_size, output_size=output_size, batch_size=batch_size, shuffle=shuffle, prefetch=prefetch)
    val_dataset = construct_dataset(x_val, y_val, input_size=input_size, output_size=output_size, batch_size=batch_size, testing=True, test_on_patches=validate_on_patches, shuffle=False, prefetch=prefetch)
    return train_dataset, val_dataset
