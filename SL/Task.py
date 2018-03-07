"""
Code written by Kevin S. Luck
2018
"""

class Task:
    """ This class is the base class for each environment (wrapper).
    """
    def __init__(self, config, render):
        """ Sets up class with config and renderer.

        Args:
            config: A dictonary which contains config-options.
            render: Boolean whihc renders environment if true.
        """
        pass

    def sample_set(self, nmbr_episodes, test =False):
        if test:
            return self.sample_from_test_set(nmbr_episodes=nmbr_episodes)
        else:
            return self.sample_from_training_set(nmbr_episodes=nmbr_episodes)

    def sample_from_training_set(self, nmbr_episodes):
        pass

    def sample_from_test_set(self, nmbr_episodes):
        pass

    def step(self, action, test, sess):
        """ Given an action the environment computes the next state.

        Args:
            action: Numpy array which represents the action for the current
                    state the environment is in.
            test:   Boolean which represents if this is test or training. True
                    for test.
            sess:   A tesnorflow session.

        Returns:
            reward: A float value with the reward of the transition performed.
            next_state: The state the environment is in after performing the
                        action
        """
        reward = None
        next_state = None
        return reward, next_state

    def is_finished(self, sess):
        """ Returns true if the current episode has ended.

        Args:
            sess: A tensorflow session object

        Returns:
            Boolean if session has ended, false if not.
        """
        return True

    def reset(self, test, sess):
        """ Reset operation for restarting the environment and starting a new
        episode.

        Args:
            test:   Boolean which represents if this is test or training. True
                    for test.
            sess:   A tesnorflow session.

        Returns:
            start_state: The initial state of the new episode
        """
        startState = None
        return startState

    def get_observation_size(self):
        """Returns the observation/state size.

        Returns:
            Tuple which contains the dimensions of the state.
        """
        pass

    def get_action_size(self):
        """Returns the size of the actions.

        Returns:
            Tuple which contains the dimenions of the action.
        """
        pass

    def initialize(self, sess):
        """ Called exactly once when the agent is set up before the very first
        episode. This function does not get called again when the graph is be
        restored from a checkpoint and returns to an episode which is not the
        very first one.
        Can be used to initialize the environment with parameters for the very
        first episode.

        Args:
            sess: A tensorflow session.

        """
        pass

    def initialize_restore(self, sess):
        """ Unlike initialize, this function gets called every time the graph
        is restored from a checkpoint and returns to exectution of episodes.

        Args:
            sess: A tensorflow session.
        """
        pass
