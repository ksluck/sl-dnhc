"""
Code written by Kevin S. Luck
2018
"""
import tensorflow as tf
import numpy as np
import warnings

def get_target_updates(vars, target_vars, tau):
    """ This code is almost identical to the OpenAI DDPG target update function.
    """
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)

def update_target_network(net, target_net, tau):
    """Creates tensorflow ops which assigns the variables of the orginal network
    to the target network with (1. - tau)

    This code is inspired by a similar function in the OpenAI DDPG agent.

    Args:
        net: Original DM sonnet network
        target_net: Target network which is a sonnet net
        tau: float value between 0.0 and 1.0

    Returns:
        One tensorflow op for the assignment of the values from net to target_net
    """
    net_vars = net.get_variables()
    target_net_vars = target_net.get_variables()

    assert len(net_vars) == len(target_net_vars)

    update_ops = []

    for var, target_var in zip(net_vars, target_net_vars):
        update_ops.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))

    return tf.group(*update_ops)

class TfEpisodeReplayBuffer():
    """ This class stores values in Tensors and is therefore restart-safe."""

    def __init__(self, size, nmbr_episode_transitions, state_size, action_size, batch_size=5,scope='Episodic_Buffer'):
        self._max_size = size
        self._ep_length = nmbr_episode_transitions + 1
        self._state_size = state_size
        self._action_size = action_size
        with tf.variable_scope(scope):
            self._buffer_states = tf.Variable(tf.zeros(
              (self._max_size, self._ep_length,) + self._state_size), dtype=tf.float32)
            self._buffer_actions = tf.Variable(tf.zeros(
              (self._max_size, self._ep_length,) + self._action_size), dtype=tf.float32)
            self._buffer_rewards = tf.Variable(tf.zeros(
              (self._max_size, self._ep_length)), dtype=tf.float32)
            self._buffer_discount = tf.Variable(tf.zeros(
              (self._max_size, self._ep_length)), dtype=tf.float32)
            self._pointer = tf.Variable(0, dtype=tf.int64)
            self._full = tf.Variable(False, dtype=tf.bool)
            self._update_states = tf.placeholder(tf.float32,
              shape=(self._ep_length,) + self._state_size)
            self._update_actions = tf.placeholder(tf.float32,
              shape=(self._ep_length,) + self._action_size)
            self._update_rewards = tf.placeholder(tf.float32,
              shape=[self._ep_length])
            self._update_discount = tf.placeholder(tf.float32,
              shape=[self._ep_length])
            self._update_buffer_states_op = tf.scatter_update(
              self._buffer_states, self._pointer, self._update_states)
            self._update_buffer_actions_op = tf.scatter_update(
              self._buffer_actions, self._pointer, self._update_actions)
            self._update_buffer_rewards_op = tf.scatter_update(
              self._buffer_rewards, self._pointer, self._update_rewards)
            self._update_buffer_discount_op = tf.scatter_update(
              self._buffer_discount, self._pointer, self._update_discount)
            self._update_op = tf.group(
              self._update_buffer_states_op,
              self._update_buffer_actions_op,
              self._update_buffer_rewards_op,
              self._update_buffer_discount_op)

            self._gather_indicies = tf.placeholder(tf.int32,
              shape=(None,))
            self._gather_states = tf.gather(self._buffer_states,self._gather_indicies)
            self._gather_actions = tf.gather(self._buffer_actions, self._gather_indicies)
            self._gather_rewards = tf.gather(self._buffer_rewards, self._gather_indicies)
            self._gather_discounts = tf.gather(self._buffer_discount, self._gather_indicies)
            self._increment_pointer = tf.assign_add(self._pointer, 1)
            self._reset_pointer = tf.assign(self._pointer, 0)
            #self._pointer_update_op = tf.cond(self._pointer < self._max_size-1,
            #  lambda: self._increment_pointer, lambda: self._reset_pointer)

            with tf.control_dependencies([self._increment_pointer]):
                self._update_full_op = tf.assign(self._full,
                  tf.logical_or(self._full,
                    tf.equal(self._pointer, self._max_size-1)))
                with tf.control_dependencies([self._update_full_op]):
                    self._pointer_update_op = tf.cond(self._pointer < self._max_size,
                      lambda: tf.identity(self._pointer), lambda: tf.assign(self._pointer, 0))

        self._curr_episode_states = np.zeros((self._ep_length,) + self._state_size)
        self._curr_episode_actions = np.zeros((self._ep_length,) + self._action_size)
        self._curr_episode_rewards = np.zeros((self._ep_length))
        self._curr_episode_discount = np.zeros((self._ep_length))
        self._curr_step = 0

    def add(self, state_t1, action, reward, discount, state_t2, end=False, sess=None):
        if self._curr_step == 0:
            self._curr_episode_states[self._curr_step,:] = state_t1
        self._curr_step += 1
        if end:
            # We have to check with -2 because of the start state
            if self._curr_step < self._ep_length - 2:
                warnings.warn('Episode smaller (length {}) than episode length ({}).'.format(self._curr_step+1, self._ep_length), UserWarning)
        else:
            if self._curr_step > self._ep_length - 2:
                raise ValueError('End of episode should have been reached.')
        self._curr_episode_states[self._curr_step,:] = state_t2
        self._curr_episode_actions[self._curr_step - 1,:] = action
        self._curr_episode_rewards[self._curr_step - 1] = reward
        self._curr_episode_discount[self._curr_step - 1] = discount
        if end:
            feed_dict = {
              self._update_states: self._curr_episode_states,
              self._update_actions: self._curr_episode_actions,
              self._update_rewards: self._curr_episode_rewards,
              self._update_discount: self._curr_episode_discount,
            }
            sess.run(self._update_op, feed_dict=feed_dict)
            sess.run(self._pointer_update_op)
            self._curr_episode_states = np.zeros((self._ep_length,) + self._state_size)
            self._curr_episode_actions = np.zeros((self._ep_length,) + self._action_size)
            self._curr_episode_rewards = np.zeros((self._ep_length))
            self._curr_episode_discount = np.zeros((self._ep_length))
            self._curr_step = 0

    def isempty(self, sess=None):
        size, full = sess.run([self._pointer, self._full])

        return size == 0 and not full

    def size(self, sess=None):
        size, full = sess.run([self._pointer, self._full])

        if full:
            return self._max_size
        else:
            return size

    def sample(self, nmbr_episodes, sess=None):
        if self.isempty(sess):
            raise Exception('Buffer is empty')
        full = sess.run(self._full)
        if full:
            size = self._max_size
        else:
            size = sess.run(self._pointer)

        indicies = range(0,size)
        indicies = np.random.choice(indicies, nmbr_episodes, replace=False)
        indicies = indicies.tolist()

        states, actions, reward, discount = sess.run([self._gather_states,
          self._gather_actions, self._gather_rewards, self._gather_discounts],
           feed_dict={self._gather_indicies: indicies})
        return states, actions, reward, discount

class TfEpisodePriorityBuffer():
    """ This class stores values in Tensors and is therefore restart-safe.
    This buffer samples episodes with a probability proportional to the ranking
    of each episode based on the cumulative reward """

    def __init__(self, size, nmbr_episode_transitions, state_size, action_size, batch_size=5,scope='Episodic_Buffer'):
        """

        Args:
            size: integer value for the number of episodes stored in the buffer
            nmbr_episode_transitions: number of transitions per episode
            state_size: Tuple defining the dimensions of the state
            action_size: Tuple defining the dimenions of the actions
            scope: String which defines the scope of the variables
        """
        self._max_size = size
        self._ep_length = nmbr_episode_transitions + 1
        self._state_size = state_size
        self._action_size = action_size
        with tf.variable_scope(scope):
            self._buffer_states = tf.Variable(tf.zeros(
              (self._max_size, self._ep_length,) + self._state_size), dtype=tf.float32)
            self._buffer_actions = tf.Variable(tf.zeros(
              (self._max_size, self._ep_length,) + self._action_size), dtype=tf.float32)
            self._buffer_rewards = tf.Variable(tf.zeros(
              (self._max_size, self._ep_length)), dtype=tf.float32)
            self._buffer_discount = tf.Variable(tf.zeros(
              (self._max_size, self._ep_length)), dtype=tf.float32)
            self._pointer = tf.Variable(0, dtype=tf.int64)
            self._full = tf.Variable(False, dtype=tf.bool)
            self._update_states = tf.placeholder(tf.float32,
              shape=(self._ep_length,) + self._state_size)
            self._update_actions = tf.placeholder(tf.float32,
              shape=(self._ep_length,) + self._action_size)
            self._update_rewards = tf.placeholder(tf.float32,
              shape=[self._ep_length])
            self._update_discount = tf.placeholder(tf.float32,
              shape=[self._ep_length])
            self._update_buffer_states_op = tf.scatter_update(
              self._buffer_states, self._pointer, self._update_states)
            self._update_buffer_actions_op = tf.scatter_update(
              self._buffer_actions, self._pointer, self._update_actions)
            self._update_buffer_rewards_op = tf.scatter_update(
              self._buffer_rewards, self._pointer, self._update_rewards)
            self._update_buffer_discount_op = tf.scatter_update(
              self._buffer_discount, self._pointer, self._update_discount)
            self._update_op = tf.group(
              self._update_buffer_states_op,
              self._update_buffer_actions_op,
              self._update_buffer_rewards_op,
              self._update_buffer_discount_op)

            self._gather_indicies = tf.placeholder(tf.int32,
              shape=(None,))
            self._gather_states = tf.gather(self._buffer_states,self._gather_indicies)
            self._gather_actions = tf.gather(self._buffer_actions, self._gather_indicies)
            self._gather_rewards = tf.gather(self._buffer_rewards, self._gather_indicies)
            self._gather_discounts = tf.gather(self._buffer_discount, self._gather_indicies)
            self._increment_pointer = tf.assign_add(self._pointer, 1)
            self._reset_pointer = tf.assign(self._pointer, 0)
            self._sum_rewards_episode = tf.reduce_sum(self._buffer_rewards, axis=1)

            #self._pointer_update_op = tf.cond(self._pointer < self._max_size-1,
            #  lambda: self._increment_pointer, lambda: self._reset_pointer)

            with tf.control_dependencies([self._increment_pointer]):
                self._update_full_op = tf.assign(self._full,
                  tf.logical_or(self._full,
                    tf.equal(self._pointer, self._max_size - 1)))
                with tf.control_dependencies([self._update_full_op]):
                    self._pointer_update_op = tf.cond(self._pointer < self._max_size,
                      lambda: tf.identity(self._pointer), lambda: tf.assign(self._pointer, 0))

        self._curr_episode_states = np.zeros((self._ep_length,) + self._state_size)
        self._curr_episode_actions = np.zeros((self._ep_length,) + self._action_size)
        self._curr_episode_rewards = np.zeros((self._ep_length))
        self._curr_episode_discount = np.zeros((self._ep_length))
        self._curr_step = 0

    def add(self, state_t1, action, reward, discount, state_t2, end=False, sess=None):
        """Adds a transition to the buffer.
        It is important that the transitions are added to the buffer one after
        each other such that state_t2 == state_t1 holds for two subsequent
        calls to add(...).

        Args:
            state_t1: State in which the transition starts
            action: Action performed
            reward: Float reward received
            discount: Discount
            state_t2: State reached after executing action in state_t1
            end: Boolean which is true for the last transition in the episode
            sess: A tensorflow session.
        """
        if self._curr_step == 0:
            self._curr_episode_states[self._curr_step,:] = state_t1
        self._curr_step += 1
        if end:
            # We have to check with -2 because of the start state
            if self._curr_step < self._ep_length - 2:
                warnings.warn('Episode smaller (length {}) than episode length ({}).'.format(self._curr_step+1, self._ep_length), UserWarning)
        else:
            if self._curr_step > self._ep_length - 2:
                raise ValueError('End of episode should have been reached.')
        self._curr_episode_states[self._curr_step,:] = state_t2
        self._curr_episode_actions[self._curr_step - 1,:] = action
        self._curr_episode_rewards[self._curr_step - 1] = reward
        self._curr_episode_discount[self._curr_step - 1] = discount
        if end:
            feed_dict = {
              self._update_states: self._curr_episode_states,
              self._update_actions: self._curr_episode_actions,
              self._update_rewards: self._curr_episode_rewards,
              self._update_discount: self._curr_episode_discount,
            }
            sess.run(self._update_op, feed_dict=feed_dict)
            sess.run(self._pointer_update_op)
            self._curr_episode_states = np.zeros((self._ep_length,) + self._state_size)
            self._curr_episode_actions = np.zeros((self._ep_length,) + self._action_size)
            self._curr_episode_rewards = np.zeros((self._ep_length))
            self._curr_episode_discount = np.zeros((self._ep_length))
            self._curr_step = 0

    def isempty(self, sess=None):
        """ True if the buffer is still isempty

        Args:
            sess: tensorflow session
        Returns:
            Booal which is true if the buffer is empty.
        """
        size, full = sess.run([self._pointer, self._full])

        return size == 0 and not full

    def size(self, sess=None):
        """ Returns the number of episodes stored in the buffer.

        Args:
            sess: A tensorflow session

        Returns:
            An integer which is equal to the number of episodes stored in the
            buffer.
        """
        size, full = sess.run([self._pointer, self._full])

        if full:
            return self._max_size
        else:
            return size

    def sample(self, nmbr_episodes, sess=None):
        """Samples n distinct number of episodes from the buffer.

        Args:
            nmbr_episodes: Number of episodes
            sess: A tensorflow session

        Returns:
            A tuple of (state, actions, reward, discount) with each entry being
            a numpy matrix/tensor with the first dimenions being the episode,
            the second the time and then the corresponding number of dimensions
        """
        if self.isempty(sess):
            raise Exception('Buffer is empty')
        full = sess.run(self._full)
        if full:
            size = self._max_size
        else:
            size = sess.run(self._pointer)

        rewards = sess.run(self._sum_rewards_episode)
        weight_strategy = 'cma-es'
        if weight_strategy == 'normal':
            rewards_min = np.amin(rewards)
            rewards = rewards + rewards_min
            weights = rewards[0:size] / np.sum(rewards[0:size])
        else:
            rewards_sort_idx = np.argsort(rewards[0:size])
            rewards_sort_idx = size - rewards_sort_idx
            weights = np.log(size + 0.5) - np.log(rewards_sort_idx)
            weights = weights / np.sum(weights)

        indicies = range(0,size)
        indicies = np.random.choice(indicies, nmbr_episodes, replace=False, p=weights)
        indicies = indicies.tolist()

        #indicies = np.random.randint(0, size, nmbr_episodes)
        #indicies = indicies.tolist()

        states, actions, reward, discount = sess.run([self._gather_states,
          self._gather_actions, self._gather_rewards, self._gather_discounts],
           feed_dict={self._gather_indicies: indicies})
        return states, actions, reward, discount
