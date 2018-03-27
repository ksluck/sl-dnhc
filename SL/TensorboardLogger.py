"""
Code written by Kevin S. Luck
2018
"""
import tensorflow as tf
import numpy as np
import SL.Logger as Logger


class TensorboardLogger(Logger.Logger):

    def __init__(self, config = None):
        """ Initializes the tensorboard logger and embeds
        it in the graph.
        Has logging verbosity 3.

        Args:
            config: Dictonary whoch contains the config parameters of the logger
                    which are: config['summaries_dir'] and config['parent_dir'].
            config['parent_dir']: Parent directory, ends with /
            config['summaries_dir']: Folder in which to save the tensorboard logs
        """
        summaries_dir = config['summaries_dir']
        summaries_dir = config['parent_dir'] + summaries_dir
        tf.logging.set_verbosity(3)
        with tf.name_scope('tensorboard_training'):
            # Eisode Reward
            self._episode_reward = tf.get_variable(
              name="episode_loss",
              shape=[],
              dtype=tf.float32,
              initializer=tf.zeros_initializer(),
              trainable=False)
            self._placeholder_ep_reward = tf.placeholder(tf.float32,
              shape=[], name='episode_reward_input')
            self._assign_reward_op = tf.assign(self._episode_reward,
              self._placeholder_ep_reward)

            with tf.name_scope('summaries'):
              tf.summary.scalar('training_loss', self._episode_reward)


        with tf.name_scope('tensorboard_test'):
            self._test_episode_reward = tf.get_variable(
              name="test_episode_loss",
              shape=[],
              dtype=tf.float32,
              initializer=tf.zeros_initializer(),
              trainable=False)
            self._test_placeholder_ep_reward = tf.placeholder(tf.float32,
              shape=[], name='test_episode_reward_input')
            self._test_assign_reward_op = tf.assign(self._test_episode_reward,
              self._test_placeholder_ep_reward)

            with tf.name_scope('summaries'):
              tf.summary.scalar('test_loss', self._test_episode_reward)

        self._merged = tf.summary.merge_all()
        self._logger_writer = tf.summary.FileWriter(summaries_dir)
        self._first_entry = True
        self._episode = 0

    def logLoss(self, nmbr_steps, episode, loss, test = False, sess = None, agent_id=0):
        """ Logs the rewards to a tensorboard file.

        Args:
          nmbr_steps: Number of steps performed this episode.
          episode: number of the current episode.
          reward: reward achieved throughout the episode.
          sess: A tensorflow session
          agent_id: ID of agent, not used currently.
        """
        tf.logging.info('Finished Episode {} with loss {}'.format(episode, loss))
        reward = loss/nmbr_steps
        if self._first_entry:
            self._logger_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=episode)
            self._logger_writer.add_graph(sess.graph)
            self._first_entry = False

        if test:
            sess.run([self._test_assign_reward_op], feed_dict = {
              self._test_placeholder_ep_reward : reward
              })
        else:
            sess.run([self._assign_reward_op], feed_dict = {
              self._placeholder_ep_reward : reward,
              })
        if episode > self._episode:
            summary = sess.run(self._merged)
            self._logger_writer.add_summary(summary, episode)
            self._logger_writer.flush()
            self._step_rewards = []
        self._episode = episode
