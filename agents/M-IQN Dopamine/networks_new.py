from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from absl import logging

import cv2
import gin
import gym
from gym.spaces.box import Box
import numpy as np
import tensorflow as tf

CARTPOLE_MIN_VALS = np.array([-2.4, -5., -math.pi/12., -math.pi*2.])
CARTPOLE_MAX_VALS = np.array([2.4, 5., math.pi/12., math.pi*2.])
ACROBOT_MIN_VALS = np.array([-1., -1., -1., -1., -5., -5.])
ACROBOT_MAX_VALS = np.array([1., 1., 1., 1., 5., 5.])


NATURE_DQN_OBSERVATION_SHAPE = (84, 84)  # Size of downscaled Atari 2600 frame.
NATURE_DQN_DTYPE = tf.uint8  # DType of Atari 2600 observations.
NATURE_DQN_STACK_SIZE = 4  # Number of frames in the state stack.

DQNNetworkType = collections.namedtuple('dqn_network', ['q_values'])
RainbowNetworkType = collections.namedtuple(
    'c51_network', ['q_values', 'logits', 'probabilities'])
ImplicitQuantileNetworkType = collections.namedtuple(
    'iqn_network', ['quantile_values', 'quantiles'])


@gin.configurable
class BasicDiscreteDomainNetwork(tf.keras.layers.Layer):
  """The fully connected network used to compute the agent's Q-values.
    This sub network used within various other models. Since it is an inner
    block, we define it as a layer. These sub networks normalize their inputs to
    lie in range [-1, 1], using min_/max_vals. It supports both DQN- and
    Rainbow- style networks.
    Attributes:
      min_vals: float, minimum attainable values (must be same shape as
        `state`).
      max_vals: float, maximum attainable values (must be same shape as
        `state`).
      num_actions: int, number of actions.
      num_atoms: int or None, if None will construct a DQN-style network,
        otherwise will construct a Rainbow-style network.
      name: str, used to create scope for network parameters.
      activation_fn: function, passed to the layer constructors.
  """

  def __init__(self, min_vals, max_vals, num_actions,
               num_atoms=None, name=None,
               activation_fn=tf.keras.activations.relu):
    super(BasicDiscreteDomainNetwork, self).__init__(name=name)
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.min_vals = min_vals
    self.max_vals = max_vals
    # Defining layers.
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                        name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(512, activation=activation_fn,
                                        name='fully_connected')
    if num_atoms is None:
      self.last_layer = tf.keras.layers.Dense(num_actions,
                                              name='fully_connected')
    else:
      self.last_layer = tf.keras.layers.Dense(num_actions * num_atoms,
                                              name='fully_connected')

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input."""
    x = tf.cast(state, tf.float32)
    x = self.flatten(x)
    x -= self.min_vals
    x /= self.max_vals - self.min_vals
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.last_layer(x)
    return 


@gin.configurable
class ImplicitQuantileNetwork_New(tf.keras.Model):
  """The Implicit Quantile Network (Dabney et al., 2018).."""

  def __init__(self, num_actions, quantile_embedding_dim, name=None):
    """Creates the layers used calculating quantile values.
    Args:
      num_actions: int, number of actions.
      quantile_embedding_dim: int, embedding dimension for the quantile input.
      name: str, used to create scope for network parameters.
    """
    super(ImplicitQuantileNetwork_New, self).__init__(name=name)
    self.num_actions = num_actions
    self.quantile_embedding_dim = quantile_embedding_dim
    # We need the activation function during `call`, therefore set the field.
    self.activation_fn = tf.keras.activations.relu
    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    
    self.net = BasicDiscreteDomainNetwork(
        CARTPOLE_MIN_VALS, CARTPOLE_MAX_VALS, num_actions)

    self.dense1 = tf.keras.layers.Dense(
        512, activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name='fully_connected')
    
    self.dense2 = tf.keras.layers.Dense(
        num_actions, kernel_initializer=self.kernel_initializer,
        name='fully_connected')

  def call(self, state, num_quantiles):
    """Creates the output tensor/op given the state tensor as input.
    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.
    Args:
      state: `tf.Tensor`, contains the agent's current state.
      num_quantiles: int, number of quantile inputs.
    Returns:
      collections.namedtuple, that contains (quantile_values, quantiles).
    """
    batch_size = state.get_shape().as_list()[0]
    x = self.net(state)

    state_vector_length = x.get_shape().as_list()[-1]
    state_net_tiled = tf.tile(x, [num_quantiles, 1])
    quantiles_shape = [num_quantiles * batch_size, 1]

    quantiles = tf.random.uniform(
        quantiles_shape, minval=0, maxval=1, dtype=tf.float32)
    quantile_net = tf.tile(quantiles, [1, self.quantile_embedding_dim])
    pi = tf.constant(math.pi)
    quantile_net = tf.cast(tf.range(
        1, self.quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
    quantile_net = tf.cos(quantile_net)
    # Create the quantile layer in the first call. This is because
    # number of output units depends on the input shape. Therefore, we can only
    # create the layer during the first forward call, not during `.__init__()`.
    if not hasattr(self, 'dense_quantile'):
      self.dense_quantile = tf.keras.layers.Dense(
          state_vector_length, activation=self.activation_fn,
          kernel_initializer=self.kernel_initializer)
    quantile_net = self.dense_quantile(quantile_net)
    x = tf.multiply(state_net_tiled, quantile_net)
    x = self.dense1(x)
    quantile_values = self.dense2(x)
    return ImplicitQuantileNetworkType(quantile_values, quantiles)
