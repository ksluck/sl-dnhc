from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt

import SL.Agent as Agent
import numpy as np

import networks.dnhc.dnc_alu as dnc

class DnhcAgent(Agent.Agent):
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
        dnc_access_instr_config = self._config['instr_access_config']
        dnc_controller_config = self._config['controller_config']
        dnc_output_size = self._output_size[0]
        dnc_clip_value = self._config['clip_value']
        self._max_grad_norm = self._config['max_grad_norm']
        optimizer_epsilon = self._config['optimizer_epsilon']
        learning_rate = self._config['learning_rate']

        #TODO
        self._episode_length = 2*4

        dnc_core = dnc.DNC(dnc_access_config, dnc_access_instr_config, dnc_controller_config, dnc_output_size, dnc_clip_value)

        self._batch_sequence = tf.placeholder(tf.float32,
          shape=[self._batch_size, self._episode_length] + list(self._observation_shape), name='input')
        self._batch_mask = tf.placeholder(tf.float32,
          shape=[self._episode_length], name='mask')
        self._batch_output = tf.placeholder(tf.float32,
          shape=[self._batch_size, self._episode_length] + list(self._output_size), name='output')

        initial_state = dnc_core.initial_state(self._batch_size)
        self.__initial_state = initial_state

        with tf.variable_scope('training_'):
            self._output_sequence, output_final_state = tf.nn.dynamic_rnn(
                cell=dnc_core,
                inputs=self._batch_sequence,
                time_major=False,
                initial_state=initial_state)

        #self._loss = tf.boolean_mask(self._output_sequence - self._batch_output, self._batch_mask, axis=1)
        #self._loss = tf.reduce_mean(tf.square(self._loss))/2.0
        output_masked = tf.boolean_mask(self._output_sequence, self._batch_mask, axis=1)
        batch_output_masked = tf.boolean_mask(self._batch_output, self._batch_mask, axis=1)
        self._loss = tf.losses.huber_loss(labels=batch_output_masked, predictions=output_masked, delta=0.5)#, reduction=tf.losses.Reduction.MEAN)
        #self._loss = self._loss / (self._batch_size * self._episode_length)

        trainable_variables = dnc_core.get_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self._loss, trainable_variables), self._max_grad_norm)

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate, epsilon=optimizer_epsilon)
        self._optim_op = optimizer.apply_gradients(zip(grads, trainable_variables))


        init_state_tmp = dnc_core.initial_state(1)
        #print(initial_state.access_state_2)
        init_state_tmp = init_state_tmp._replace(access_state_2=init_state_tmp.access_state_2._replace(memory=tf.expand_dims(initial_state.access_state_2.memory[0,:,:], axis=0)))
        init_state_step_flat = tf.contrib.framework.nest.flatten(init_state_tmp)
        with tf.name_scope('init_state_vars'):
            init_state_step_vars = [tf.Variable(initial_value=x, trainable=False)
                                     for x in init_state_step_flat]
        init_state_step_vars_tuple = tf.contrib.framework.nest.pack_sequence_as(init_state_tmp,
                                                     init_state_step_vars)
        self._init_state_step_vars_tuple = init_state_step_vars_tuple

        # RNN module for making only one step
        # Technically this is unneccessary
        # Input to RNN module for a single step
        self._input_sequence_step = tf.placeholder(tf.float32,
          shape=[1, 1] + list(self._observation_shape), name='input_step')
        #with tf.name_scope('Dynamic_step'):
        with tf.variable_scope('step_'):
            self._output_sequence_step, output_state_step = tf.nn.dynamic_rnn(
                cell=dnc_core,
                inputs=self._input_sequence_step,
                time_major=False,
                initial_state=init_state_step_vars_tuple,
                scope="rnn_step")
        self._output_state_step = output_state_step

        # Flatten the output state and generate ops to assign output state to initial state
        output_state_flat = tf.contrib.framework.nest.flatten(output_state_step)
        with tf.control_dependencies(output_state_flat):
            output_state_assign_ops = [tf.assign(var,val) for var,val in zip(init_state_step_vars, output_state_flat)]
            self._assign_last_state_to_current_op = tf.group(*output_state_assign_ops)
        # Op to asign the initial state to the input_state variables
        init_state_assign_ops = [tf.assign(var,val) for var,val in zip(init_state_step_vars,init_state_step_flat)]
        self._assign_initial_state_to_current_op = tf.group(*init_state_assign_ops)


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
            print("input")
            print(inputs[0])
            output, loss = sess.run([self._output_sequence,self._loss], feed_dict=feed_dict)
            print("Output")
            print(output[0])
        else:
            loss, _ = sess.run([self._loss, self._optim_op], feed_dict=feed_dict)
        return loss

    def step(self, state, reward, episode, step, test, sess):
        # state - current state
        # reward - reward from previous action
        feed_dict = {
            self._input_sequence_step : [[state]],
        }

        action, dnc_state, _, dnc_training_init = sess.run([self._output_sequence_step, self._output_state_step, self._assign_last_state_to_current_op, self.__initial_state], feed_dict=feed_dict)
        self._mem_state = dnc_state.access_state.memory
        self._instr_mem = dnc_state.access_state_2.memory
        self._instr_mem_check = dnc_training_init.access_state_2.memory
        #action = action[0][0]
        #sess.run()
        return action

    def end_episode(self, final_state, reward, test, sess):
        pass

    def start_episode(self, start_state, test, sess):
        sess.run(self._assign_initial_state_to_current_op)
        pass

    def initialize(self, sess):
        pass

    def initialize_restore(self, sess):
        pass
