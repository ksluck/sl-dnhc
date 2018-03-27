from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt

import SL.Agent as Agent
import numpy as np

import networks.dnhc.dnc_original as dnc

class DncAgent(Agent.Agent):
    def __init__(self, env, config):
        self._observation_size = env.get_observation_size()
        self._action_size = env.get_action_size()
        self._config = config

        self._observation_shape = env.get_observation_size()
        self._output_size = env.get_action_size()

        self._batch_size = 32

        self._build()


    def _build(self):
        dnc_access_config = self._config['data_access_config']
        dnc_controller_config = self._config['controller_config']
        dnc_output_size = self._output_size[0]
        dnc_clip_value = self._config['clip_value']
        self._max_grad_norm = self._config['max_grad_norm']
        optimizer_epsilon = self._config['optimizer_epsilon']
        learning_rate = self._config['learning_rate']

        #TODO
        self._episode_length = 2*4

        dnc_core = dnc.DNC(dnc_access_config, dnc_controller_config, dnc_output_size, dnc_clip_value)

        self._batch_sequence = tf.placeholder(tf.float32,
          shape=[self._batch_size, self._episode_length] + list(self._observation_shape), name='input')
        self._batch_mask = tf.placeholder(tf.float32,
          shape=[self._episode_length], name='mask')
        self._batch_output = tf.placeholder(tf.float32,
          shape=[self._batch_size, self._episode_length] + list(self._output_size), name='output')

        initial_state = dnc_core.initial_state(self._batch_size)

        self._output_sequence, output_final_state = tf.nn.dynamic_rnn(
            cell=dnc_core,
            inputs=self._batch_sequence,
            time_major=False,
            initial_state=initial_state)

        self._loss = tf.boolean_mask(self._output_sequence - self._batch_output, self._batch_mask, axis=1)
        self._loss = tf.reduce_mean(tf.square(self._loss))/2.0
        trainable_variables = dnc_core.get_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self._loss, trainable_variables), self._max_grad_norm)

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate, epsilon=optimizer_epsilon)
        self._optim_op = optimizer.apply_gradients(zip(grads, trainable_variables))


    def evaluate_batch(self, inputs, expected_outputs, test, sess):
        """
        Returns:
            a loss
        """
        feed_dict = {
            self._batch_sequence: inputs,
            self._batch_mask: inputs[1,:,3],
            self._batch_output: expected_outputs,
        }

        if test:
            loss = sess.run(self._loss, feed_dict=feed_dict)
        else:
            loss, _ = sess.run([self._loss, self._optim_op], feed_dict=feed_dict)
        return loss

    def step(self, state, reward, episode, step, test, sess):
        # state - current state
        # reward - reward from previous action
        action = np.random.randn(*self._action_size)/3.0
        action = np.minimum(np.ones(self._action_size), np.maximum(-np.ones(self._action_size),action))
        return action

    def end_episode(self, final_state, reward, test, sess):
        pass

    def start_episode(self, start_state, test, sess):
        pass

    def initialize(self, sess):
        pass

    def initialize_restore(self, sess):
        pass
