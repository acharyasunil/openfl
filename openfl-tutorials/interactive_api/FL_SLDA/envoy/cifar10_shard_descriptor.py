# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CIFAR10 Shard Descriptor (using `tf.data.Dataset` API)"""
import logging
from typing import List, Tuple

import tensorflow as tf

import tensorflow_datasets as tfds
import logging
from typing import Any, List, Mapping, MutableMapping, Optional

import numpy as np
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

from synthesizer import synthesize_by_sharding_over_labels
from utils import as_tuple, decode_example, get_label_distribution
from config import Decoders, IMG_AUGMENT_LAYERS

logger = logging.getLogger(__name__)

IMG_SIZE = (32, 32)
BATCH_SIZE = 128
SHUFFLE_BUFFER = 16384

def extract_features(dataset: tf.data.Dataset, model: Any) -> tf.data.Dataset:
  """Extract feature embeddings from the model for each image in the dataset.

  Args:
      dataset (tf.data.Dataset):
        A `tf.data.Dataset` instance with raw tensor image keyed as `image` and `label` as label.
      model (Any):
        A callable of type `tf.keras.Sequential` or `tf.keras.Model` or equivalent that can take `image`
        batch as input and return feature embedding.

  Returns:
      A `tf.data.Dataset` instance with each sample being a `(feature_embedding, label)` tuple
  """
  features = model.predict(dataset, verbose=1)
  labels = np.array(list(dataset.map(lambda x, y: y).unbatch().as_numpy_iterator()))
  return tf.data.Dataset.from_tensor_slices({
    'image': features,
    'label': labels
  })

class CIFAR10ShardDescriptor(ShardDescriptor):
    """
    CIFAR100 Shard Descriptor

    This example is based on `tf.data.Dataset` pipelines.
    Note that the ingestion of any model/task requires an iterable dataloader.
    Hence, it is possible to utilize these pipelines without explicit need of a
    new interface.
    """

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        """Download/Prepare CIFAR10 dataset"""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

        print("CIFAR10 Shard Descriptor Called!")

        self.prepare_and_partition_dataset()

        print("__init__ Created train and valid split : SUCCESS")
    
    def get_feature_extractor(self):

        """Choose Model backbone to extract features"""
        backbone = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights='imagenet',
            input_shape=(*IMG_SIZE, 3),
            pooling='avg'
        )
        backbone.trainable = False

        """Add augmentation/input layers"""
        feature_extractor = tf.keras.Sequential([
            tf.keras.layers.InputLayer(backbone.input_shape[1:]),
            IMG_AUGMENT_LAYERS,
            backbone,
        ], name='feature_extractor')

        return feature_extractor

    
    def prepare_and_partition_dataset(self):
        """
        Load CIFAR10 as `tf.data.Dataset`.

        Provide `rank` and `worldsize` to auto-split uniquely for each client
        for simulation purposes.
        """

        (raw_train_ds, raw_test_ds), ds_info = tfds.load('cifar10',
                                                     split=['train', 'test'],
                                                     with_info=True,
                                                     decoders=Decoders.SIMPLE_DECODER)

        print("download_and_prepare raw train and test ds split : SUCCESS")
        

        """Extract train/test feature embeddings"""
        feature_extractor = self.get_feature_extractor()


        print(f'Extracting train set features')
        train_features = extract_features(dataset=(raw_train_ds
                                                .map(decode_example(IMG_SIZE))
                                                .map(as_tuple(x='image', y='label'))
                                                .batch(BATCH_SIZE)
                                                .prefetch(tf.data.AUTOTUNE)), model=feature_extractor)
        print(f'Extracting test set features')
        test_features = extract_features(dataset=(raw_test_ds
                                                .map(decode_example(IMG_SIZE))
                                                .map(as_tuple(x='image', y='label'))
                                                .batch(BATCH_SIZE)
                                                .prefetch(tf.data.AUTOTUNE)), model=feature_extractor)
        print("Features Dataset spec")
        print('Features Dataset spec: ', train_features.element_spec)

        partitioned_dataset = synthesize_by_sharding_over_labels(train_features, 
                                                         num_partitions=self.worldsize,
                                                         shuffle_labels=True)

        train_ds = (partitioned_dataset[self.rank - 1]
            .cache()
            .shuffle(SHUFFLE_BUFFER)
            .map(as_tuple(x='image', y='label'))
            .batch(1)  # SLDA learns 1-sample at a time. Inference can be done on batch.
            .prefetch(tf.data.AUTOTUNE))


        valid_ds = (test_features
            .cache()
            .map(as_tuple(x='image', y='label'))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))
                    
        

        self._sample_shape = ds_info.features['image'].shape
        self._target_shape = tf.expand_dims(ds_info.features['label'].shape, -1).shape[1:]
        self.train_len = len(list(partitioned_dataset[self.rank - 1]))
        self.test_len = len(raw_test_ds)

        self.splits = {
            'train': train_ds,
            'valid': valid_ds,
            'train_size': self.train_len,
            'test_size': self.test_len

        }
        print("Prepare Partition : Success")
        

    def get_shard_dataset_types(self) -> List[str]:
        """Get available split names"""
        return list(self.splits)

    def get_split(self, name: str) -> tf.data.Dataset:
        """Return a shard dataset by type."""
        if name not in self.splits:
            raise Exception(f'Split name `{name}` not found.'
                            f' Expected one of {list(self.splits.keys())}')
        return self.splits[name]
                    
    @property
    def sample_shape(self) -> List[str]:
        """Return the sample shape info."""
        return list(map(str, self._sample_shape))

    @property
    def target_shape(self) -> List[str]:
        """Return the target shape info."""
        return list(map(str, self._target_shape))

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'CIFAR10 dataset, shard number {self.rank}/{self.worldsize}.'
                f'\nSamples [Train/Valid]: [{self.train_len}/{self.test_len}]')