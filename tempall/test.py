import jax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import ml_collections
import optax
import tensorflow_datasets as tfds
import logging


@jax.jit
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)

def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
        
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy

def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> train_state.TrainState:
    """Execute model training and evaluation loop.
    Args:
        config: Hyperparameter configuration for training and evaluation.
        workdir: Directory where the tensorboard summaries are written to.
    Returns:
        The train state (which includes the `.params`).
    """
    
    train_ds, test_ds = get_datasets()
    rng = jax.random.PRNGKey(0)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(state, train_ds,
                                                        config.batch_size,
                                                        input_rng)
        _, test_loss, test_accuracy = apply_model(state, test_ds['image'],
                                                  test_ds['label'])

        logging.info(
            'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
            % (epoch, train_loss, train_accuracy * 100, test_loss,
               test_accuracy * 100))

        summary_writer.scalar('train_loss', train_loss, epoch)
        summary_writer.scalar('train_accuracy', train_accuracy, epoch)
        summary_writer.scalar('test_loss', test_loss, epoch)
        summary_writer.scalar('test_accuracy', test_accuracy, epoch)

    summary_writer.flush()
    return state

def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds

class CNN(nn.Module):
    """A simple CNN model."""
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params'].unfreeze()
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)

def _get_weights_dict(obj, prefix='', suffix=''):
    """
    Get the dictionary of weights.

    Parameters
    ----------
    obj : Model or Optimizer
        The target object that we want to get the weights.

    Returns
    -------
    dict
        The weight dictionary.
    """
    weights_dict = dict()
    delim = '.'
    for layer_name, param_obj in obj.items():
        for param_name, value in param_obj.items():
            key = delim.join(filter(None, [prefix, layer_name, param_name, suffix]))
            weights_dict[key] = value

    return weights_dict

def _set_weights_dict(obj, weights_dict, prefix=''):
    """Set the object weights with a dictionary.

    The obj can be a model or an optimizer.

    Args:
        obj (Model or Optimizer): The target object that we want to set
        the weights.
        weights_dict (dict): The weight dictionary.

    Returns:
        None
    """
    delim = '.'

    if prefix == 'opt':
        model_state_dict = obj.opt_state[0][0]
    else:
        model_state_dict = obj.params

    for layer_name, param_obj in model_state_dict.items():
        for param_name, value in param_obj.items():
            key = delim.join(filter(None, [prefix, layer_name, param_name]))
            if key in weights_dict:
                model_state_dict[layer_name][param_name] = weights_dict[key] + 100

# Get Tensor
editor_relpaths = ('configs/default.py', 'train.py')

from configs import default as config_lib
config = config_lib.get_config()

config.num_epochs = 3
models = {}
for momentum in (0.8, 0.9, 0.95):
    name = f'momentum={momentum}'
    config.momentum = momentum
    state = train_and_evaluate(config, workdir=f'./models/{name}')
    models[name] = state.params
    break


model = state
suffix = ''

model_params = jax.tree_util.tree_map(np.array, model.params)
params_dict = _get_weights_dict(model_params, 'param', suffix)

if model.opt_state is not None:
    model_opt_state = jax.tree_util.tree_map(np.array, model.opt_state)[0][0]
    opt_dict = _get_weights_dict(model_opt_state, 'opt', suffix)
    params_dict.update(opt_dict)


tensor_dict = params_dict
# Model Weights
prefix = 'param'
delim = '.'

tensor_dict = jax.tree_util.tree_map(jnp.array, tensor_dict)

_set_weights_dict(model, tensor_dict, 'param')

if model.opt_state:
    _set_weights_dict(model, tensor_dict, 'opt')    

print(model)