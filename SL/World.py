"""
Code written by Kevin S. Luck
2018
"""
from RL.Agent import Agent
from RL.Environment import Environment
import threading
import tensorflow as tf
import numpy as np


class World():
    """ Class which executes test and trianing episodes and handles
    environments, agent and logger.
    """
    def __init__(self, agent, env, logger, agent_id, config):
        """
        Args:
            agent: Agent class
            env: Environments
            logger: Logger
            agent_id: Id of the agent (might be removed)
            config: Dictonary which contains the config
        """
        if not isinstance(agent, Agent):
            raise TypeError('RL is not derived from RL base class')

        if not isinstance(env, Environment):
            raise TypeError('Environment is not derived from Environment base class')

        self._agent = agent
        self._env = env
        self._logger = logger
        self._agent_id = agent_id
        self._config = config

    def execute(self, episodes=100000):
        """
        Executes n number of test and training episodes.
        Args:
            episodes: Number of episodes to execute
        """
        if episodes < 1:
            episodes = 5

        counter = 0
        step_counter = 0

        global_step = tf.get_variable(
          name="global_step",
          shape=[],
          dtype=tf.int64,
          initializer=tf.zeros_initializer(),
          trainable=False,
          collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

        global_step_incr_op = tf.assign_add(global_step,1,
          name='global_step_increment')

        saver = tf.train.Saver()

        if self._config['checkpoint_interval'] > 0:
            hooks = [
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=self._config['checkpoint_dir'],
                    save_steps=self._config['checkpoint_interval'],
                    saver=saver)
            ]
        else:
            hooks = []

        print(hooks)
        with tf.train.SingularMonitoredSession(
          hooks=hooks, checkpoint_dir=self._config['checkpoint_dir']) as sess:
          episode_counter = tf.train.global_step(sess, global_step)

          start_iteration = tf.train.global_step(sess, global_step)
          if start_iteration == 0:
              self._agent.initialize(sess)
              self._env.initialize(sess)
          else:
              self._agent.initialize_restore(sess)
              self._env.initialize_restore(sess)


          for episode_counter in range(start_iteration, episodes):
              #Training
              inputs, expected_outputs = self._env.sample_set(nmbr_episodes=32, test=False)
              loss = self._agent.evaluate_batch(inputs=inputs, expected_outputs=expected_outputs, test=False, sess=sess)
              self._logger.logLoss(nmbr_steps=1, episode=episode_counter, loss=loss, test=False sess=sess)

              #test
              inputs, expected_outputs = self._env.sample_set(nmbr_episodes=32, test=True)
              loss = self._agent.evaluate_batch(inputs=inputs, expected_outputs=expected_outputs, test=True, sess=sess)
              self._logger.logLoss(nmbr_steps=1, episode=episode_counter, loss=loss, test=True, sess=sess)

              sess.run(global_step_incr_op)


          # for episode_counter in range(start_iteration, episodes):
          #   # TODO Does it make sense to also return the first reward of the
          #   # initial state?
          #
          #   # Training
          #   state = self._env.reset(test = False, sess=sess)
          #   self._agent.start_episode(state, test = False, sess=sess)
          #   episodic_reward = 0.0
          #   steps_counter = 0
          #   reward = 0
          #
          #   while not self._env.is_finished(sess=sess):
          #       action = self._agent.step(state, reward, test = False, episode = episode_counter, step =steps_counter, sess=sess)
          #       reward, state = self._env.step(action,test = False, sess=sess)
          #       episodic_reward += reward
          #       steps_counter += 1
          #       self._logger.logStepReward(step_counter, reward,
          #         test = False, sess=sess, agent_id=self._agent_id)
          #
          #   self._agent.end_episode(state, reward, test = False, sess=sess)
          #   self._logger.logEpisodeReward(step_counter, episode_counter,
          #     episodic_reward, test = False, sess=sess, agent_id=self._agent_id)
          #
          #   # Test
          #   state = self._env.reset(test = True, sess=sess)
          #   self._agent.start_episode(state, test = True, sess=sess)
          #   episodic_reward = 0.0
          #   steps_counter = 0
          #   reward = 0
          #
          #   while not self._env.is_finished(sess=sess):
          #       action = self._agent.step(state, reward, episode = episode_counter, step =steps_counter, test = True, sess=sess)
          #       reward, state = self._env.step(action, test = True, sess=sess)
          #       episodic_reward += reward
          #       steps_counter += 1
          #       self._logger.logStepReward(step_counter, reward, test = True,
          #         sess=sess, agent_id=self._agent_id)
          #
          #   self._agent.end_episode(state, reward, test = True, sess=sess)
          #   self._logger.logEpisodeReward(step_counter, episode_counter,
          #     episodic_reward, test = True, sess=sess, agent_id=self._agent_id)
          #
          #   sess.run(global_step_incr_op)


def _execute_agent(agents, envs, logger, agent_id = 0, iterations = 1000):
    world = World(agents[agent_id], envs[agent_id], logger, agent_id=agent_id)
    world.execute(iterations)
