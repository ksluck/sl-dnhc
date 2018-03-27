# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DNC Cores.

These modules create a DNC core. They take input, pass parameters to the memory
access module, and integrate the output of memory to form an output.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf

import networks.dnhc.access as access
import networks.dnhc.access_instruction as access_instruction

DNCState = collections.namedtuple('DNCState', ('access_output', 'access_state',
                                               #'access_output_2',
                                               'access_state_2',
                                               'controller_state'))


class DNC(snt.RNNCore):
  """DNC core module.

  Contains controller and memory access module.
  """

  def __init__(self,
               access_config,
               access_config_2,
               controller_config,
               output_size,
               clip_value=None,
               name='dnc'):
    """Initializes the DNC core.

    Args:
      access_config: dictionary of access module configurations.
      controller_config: dictionary of controller (LSTM) module configurations.
      output_size: output dimension size of core.
      clip_value: clips controller and core output values to between
          `[-clip_value, clip_value]` if specified.
      name: module name (default 'dnc').

    Raises:
      TypeError: if direct_input_size is not None for any access module other
        than KeyValueMemory.
    """
    super(DNC, self).__init__(name=name)

    with self._enter_variable_scope():
      #self._controller = snt.LSTM(**controller_config)
      self._controller = snt.DeepRNN([snt.Linear(controller_config['hidden_size'])], skip_connections=False)
      self._access = access.MemoryAccess(**access_config)
      #self._access_2 = access.MemoryAccess(**access_config_2)
      self._access_2 = access_instruction.MemoryAccessInst(**access_config_2)

    self._access_output_size = np.prod(self._access.output_size.as_list())
    # self._access_output_size_2 = np.prod(self._access_2.output_size.as_list())
    self._output_size = output_size
    self._clip_value = clip_value or 0

    self._output_size = tf.TensorShape([output_size])
    self._state_size = DNCState(
        access_output=self._access_output_size,
        access_state=self._access.state_size,
        # access_output_2=self._access_output_size,
        access_state_2=self._access_2.state_size,
        controller_state=self._controller.state_size)

    self._b_init = tf.zeros(access_config_2['num_reads'])
    #self._W_init = tf.zeros(self._output_size.as_list()[0]+[])

  def _clip_if_enabled(self, x):
    if self._clip_value > 0:
      return tf.clip_by_value(x, -self._clip_value, self._clip_value)
    else:
      return x

  def _build(self, inputs, prev_state):
    """Connects the DNC core into the graph.

    Args:
      inputs: Tensor input.
      prev_state: A `DNCState` tuple containing the fields `access_output`,
          `access_state` and `controller_state`. `access_state` is a 3-D Tensor
          of shape `[batch_size, num_reads, word_size]` containing read words.
          `access_state` is a tuple of the access module's state, and
          `controller_state` is a tuple of controller module's state.

    Returns:
      A tuple `(output, next_state)` where `output` is a tensor and `next_state`
      is a `DNCState` tuple containing the fields `access_output`,
      `access_state`, and `controller_state`.
    """

    prev_access_output = prev_state.access_output
    prev_access_state = prev_state.access_state
    #prev_access_output_2 = prev_state.access_output_2
    prev_access_state_2 = prev_state.access_state_2
    prev_controller_state = prev_state.controller_state

    batch_flatten = snt.BatchFlatten()
    controller_input = tf.concat(
        [batch_flatten(inputs), batch_flatten(prev_access_output)], 1)

    controller_output, controller_state = self._controller(
        controller_input, prev_controller_state)

    controller_output = self._clip_if_enabled(controller_output)
    controller_state = snt.nest.map(self._clip_if_enabled, controller_state)

    access_output, access_state = self._access(controller_output,
                                               prev_access_state)
    access_output_2, access_state_2 = self._access_2(controller_output,
                                               prev_access_state_2)

    output = tf.concat([controller_output, batch_flatten(access_output)], 1)
    # output = snt.Linear(
    #     output_size=self._output_size.as_list()[0],
    #     name='output_linear')(output)
    with tf.name_scope('output_linear_fancy') as scope:
        weights = tf.nn.softmax(access_output_2)

        alu_1 = snt.Linear(3, name='alu_1')(output)
        alu_2 = snt.Linear(3, name='alu_2')(output)
        alu_3 = snt.Linear(3, name='alu_3')(output)

        weighted_alu_1 = tf.multiply(alu_1, weights[:,:,0])
        weighted_alu_2 = tf.multiply(alu_2, weights[:,:,1])
        weighted_alu_3 = tf.multiply(alu_3, weights[:,:,2])

        output = weighted_alu_1 + weighted_alu_2 + weighted_alu_3

    #output = tf.nn.relu(output)
    #output = snt.Linear(3, name='last_layer')(output)

    output = self._clip_if_enabled(output)

    return output, DNCState(
        access_output=access_output,
        access_state=access_state,
        # access_output_2=access_output,
        access_state_2=access_state_2,
        controller_state=controller_state)

  def initial_state(self, batch_size, dtype=tf.float32):
    flat_state_size = snt.nest.flatten(self._access_2.state_size)
    flat_initializer = [tf.random_uniform_initializer(minval=-0.1, maxval=0.1)] * len(flat_state_size)
    trainable_initializers = snt.nest.pack_sequence_as(
              structure=self._access_2.state_size, flat_sequence=flat_initializer)
    return DNCState(
        controller_state=self._controller.initial_state(batch_size, dtype),
        access_state=self._access.initial_state(batch_size, dtype),
        access_state_2=self._access_2.initial_state(batch_size, dtype, trainable=True,
                                                    trainable_initializers=trainable_initializers),
        #access_state_2=self._access_2.initial_state(batch_size, dtype),
        access_output=tf.zeros(
            [batch_size] + self._access.output_size.as_list(), dtype))

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size
