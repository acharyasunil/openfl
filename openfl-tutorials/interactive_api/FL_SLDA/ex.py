# import logging
# from typing import List, Tuple

# import tensorflow as tf

# import tensorflow_datasets as tfds
# from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

# from openfl.cl.lib.dataset.synthesizer import synthesize_by_sharding_over_labels
# from openfl.cl.lib.dataset.utils import as_tuple, decode_example, get_label_distribution
# from openfl.cl.config import Decoders, IMG_AUGMENT_LAYERS
# from openfl.cl.models.utils import extract_features
# from openfl.cl.models.slda import SLDA

# IMG_SIZE = (32, 32)
# BATCH_SIZE = 64
# SHUFFLE_BUFFER = 16384
# rank = 1
# worldsize = 3

# def get_feature_extractor():

#     """Choose Model backbone to extract features"""
#     backbone = tf.keras.applications.EfficientNetV2B0(
#         include_top=False,
#         weights='imagenet',
#         input_shape=(*IMG_SIZE, 3),
#         pooling='avg'
#     )
#     backbone.trainable = False

#     """Add augmentation/input layers"""
#     feature_extractor = tf.keras.Sequential([
#         tf.keras.layers.InputLayer(backbone.input_shape[1:]),
#         IMG_AUGMENT_LAYERS,
#         backbone,
#     ], name='feature_extractor')

#     return feature_extractor

# (raw_train_ds, raw_test_ds), ds_info = tfds.load('cifar10',
#                                                      split=['train', 'test'],
#                                                      with_info=True,
#                                                      decoders=Decoders.SIMPLE_DECODER)

# print("download_and_prepare raw train and test ds split : SUCCESS")


# """Extract train/test feature embeddings"""
# feature_extractor = get_feature_extractor()


# print(f'Extracting train set features')
# train_features = extract_features(dataset=(raw_train_ds
#                                         .map(decode_example(IMG_SIZE))
#                                         .map(as_tuple(x='image', y='label'))
#                                         .batch(BATCH_SIZE)
#                                         .prefetch(tf.data.AUTOTUNE)), model=feature_extractor)
# print(f'Extracting test set features')
# test_features = extract_features(dataset=(raw_test_ds
#                                         .map(decode_example(IMG_SIZE))
#                                         .map(as_tuple(x='image', y='label'))
#                                         .batch(BATCH_SIZE)
#                                         .prefetch(tf.data.AUTOTUNE)), model=feature_extractor)
# print("Features Dataset spec")
# print('Features Dataset spec: ', train_features.element_spec)

# partitioned_dataset = synthesize_by_sharding_over_labels(train_features, 
#                                                     num_partitions=worldsize,
#                                                     shuffle_labels=True)
# for pid in partitioned_dataset:
#     print(pid)

# train_ds = (partitioned_dataset[rank - 1]
#             .cache()
#             .shuffle(SHUFFLE_BUFFER)
#             .map(as_tuple(x='image', y='label'))
#             .batch(1)  # SLDA learns 1-sample at a time. Inference can be done on batch.
#             .prefetch(tf.data.AUTOTUNE))


# valid_ds = (test_features
#             .cache()
#             .map(as_tuple(x='image', y='label'))
#             .batch(BATCH_SIZE)
#             .prefetch(tf.data.AUTOTUNE))



# splits = {
#     'train': train_ds,
#     'valid': valid_ds
# }

# _sample_shape = ds_info.features['image'].shape
# _target_shape = tf.expand_dims(ds_info.features['label'].shape, -1).shape[1:]
# train_len = len(list(partitioned_dataset[rank - 1]))
# test_len = len(raw_test_ds)




# # Model

# model = SLDA(n_components=feature_extractor.output_shape[-1],
#              num_classes=ds_info.features['label'].num_classes)

# # model.compile(metrics=['accuracy'])

# print(model.evaluate(splits['train']))

# model.fit(splits['train'], epochs=1)

# val = model.evaluate(splits['valid'])

# print("Test DS", val)



from tensorflow import keras
import tensorflow as tf
import cloudpickle
from openfl.cl.models.slda import SLDA

filename = './workspace/model_obj.pkl'

with open(filename, 'rb') as f:
    mi = cloudpickle.load(f)

print(type(mi.model))
print(mi.model.weights)
mi.model.__class__ = SLDA
print(type(mi.model))

mi.model.compile(metrics=['accuracy'])

print(mi.model.get_weights())

mi.model.fit(tf.random.uniform((20, 1280)))

# mi.model.fit(tf.random.uniform((1, 1280)), tf.random.uniform((1,), minval=0, maxval=10, dtype=tf.int64))

val = mi.model.evaluate(tf.random.uniform((32, 1280)), tf.random.uniform((32,), minval=0, maxval=10, dtype=tf.int64))
# val = mi.model.evaluate(tf.random.uniform((32, 1280)))

print(val)
