# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom model numpy adapter."""
import jax
import numpy as np
import jax.numpy as jnp

from openfl.plugins.frameworks_adapters.framework_adapter_interface import (
    FrameworkAdapterPluginInterface,
)

class CustomFrameworkAdapter(FrameworkAdapterPluginInterface):
    """Framework adapter plugin class."""

    @staticmethod
    def get_tensor_dict(model, optimizer=None, suffix=''):
        """
        Extract tensor dict from a model and an optimizer.

        Returns:
        dict {weight name: numpy ndarray}

        Steps:
        1. Map JAX leaf params and optimizer state to numpy.
        2. flatten the state dict to {w: value}
        3. optimizer parameter always remains None?
        """

        model_params = jax.tree_util.tree_map(np.array, model.params)
        params_dict = _get_weights_dict(model_params, 'param', suffix)

        if model.opt_state:
            model_opt_state = jax.tree_util.tree_map(np.array, model.opt_state)[0][0]
            opt_dict = _get_weights_dict(model_opt_state, 'opt', suffix)
            params_dict.update(opt_dict)

        return params_dict

    @staticmethod
    def set_tensor_dict(model, tensor_dict, optimizer=None, device='cpu'):
        """
        Set the model weights with a tensor dictionary.

        Args:
            tensor_dict: the tensor dictionary
            with_opt_vars (bool): True = include the optimizer's status.
        """
        
        tensor_dict = jax.tree_util.tree_map(jnp.array, tensor_dict)

        _set_weights_dict(model, tensor_dict, 'param')

        if model.opt_state:
            _set_weights_dict(model, tensor_dict, 'opt')            

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
                model_state_dict[layer_name][param_name] = weights_dict[key]


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