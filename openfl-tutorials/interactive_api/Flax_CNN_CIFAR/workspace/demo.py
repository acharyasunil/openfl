import tensorflow_datasets as tfds
import tensorflow_datasets as tfds
import jax.numpy as jnp
import tensorflow as tf
rank=1
worldsize=4

dataset_builder = tfds.builder('cifar10')
dataset_builder.download_and_prepare()

datasets, dataset_info = dataset_builder.as_dataset(), dataset_builder.info
train_shard_size = int(len(datasets['train']) / worldsize)
test_shard_size = int(len(datasets['test']) / worldsize)

train_ds = dataset_builder.as_dataset(split=f'train[{train_shard_size * (rank - 1)}:{train_shard_size * rank}]', batch_size=-1)
test_ds = dataset_builder.as_dataset(split=f'test[{test_shard_size * (rank - 1)}:{test_shard_size * rank}]', batch_size=-1)
train_ds['image'] = jnp.float32(train_ds['image']) / 255.
test_ds['image'] = jnp.float32(test_ds['image']) / 255.

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# # Split
# x_train, y_train = x_train[rank-1::worldsize], y_train[rank-1::worldsize]
# x_test, y_test = x_test[rank-1::worldsize], y_test[rank-1::worldsize]

# train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# train_ds.element_spec[1].shape

# print(len(train_ds), train_ds, train_ds.element_spec[0].shape, train_ds.element_spec[1].shape)
print(len(train_ds['label']), train_ds['image'].shape[1:], tf.expand_dims(train_ds['label'], -1).shape[1:])