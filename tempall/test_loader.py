from distutils.log import info
import logging
from typing import List, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as jnp




# train_ds = tfds.load('cifar10', split=tfds.Split.TRAIN)
# train_ds = train_ds.map(lambda x: {'image': tf.cast(x['image'], tf.float32) / 255.,
#                                      'label': tf.cast(x['label'], tf.int32)})



# print(type(train_ds))
# print(train_ds)






###############################
rank = 1
worldsize = 2

dataset_builder = tfds.builder('cifar10')
dataset_builder.download_and_prepare()

datasets, dataset_info = dataset_builder.as_dataset(), dataset_builder.info
train_shard_size = int(len(datasets['train']) / worldsize)
test_shard_size = int(len(datasets['test']) / worldsize)

# train_ds = tfds.as_numpy(dataset_builder.as_dataset(split=f'train[{train_shard_size * (rank - 1)}:{train_shard_size * rank}]', batch_size=-1))
# test_ds = tfds.as_numpy(dataset_builder.as_dataset(split=f'test[{test_shard_size * (rank - 1)}:{test_shard_size * rank}]', batch_size=-1))
train_ds = dataset_builder.as_dataset(split=f'train[{train_shard_size * (rank - 1)}:{train_shard_size * rank}]', batch_size=-1)
test_ds = dataset_builder.as_dataset(split=f'test[{test_shard_size * (rank - 1)}:{test_shard_size * rank}]', batch_size=-1)
train_ds['image'] = jnp.float32(train_ds['image']) / 255.
train_ds['label'] = jnp.int32(train_ds['label'])
test_ds['image'] = jnp.float32(test_ds['image']) / 255.
test_ds['label'] = jnp.int32(test_ds['label'])

# _sample_shape = train_ds['image'].shape[1:]
# _target_shape = tf.expand_dims(train_ds['label'], -1).shape[1:]

splits = {
    'train': train_ds,
    'valid': test_ds
}

print(train_ds)
print(len(train_ds['image']))
# print("================================================")
# print(test_ds)



# # # His Implememtation

# # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# # # Split
# # x_train, y_train = x_train[rank-1::worldsize], y_train[rank-1::worldsize]
# # x_test, y_test = x_test[rank-1::worldsize], y_test[rank-1::worldsize]

# # train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# # test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# # print(test_ds)