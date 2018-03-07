"""
Code written by Kevin S. Luck
2018
"""
class Agent:
    """
    Base class for any agent class. Agents usually implement one or more
    network classes and compute actions to be taken in the environment.
    """
    def __init__(self, env, config):
        """
        Initializes the class and usually calls a _build class for
        further initialization.

        Args:
            env: The environment the agent acts on. Is used for requesting
                 required parameters such as action size etc.
            config: A dictonary whoch contains additional parameters required
        """
        self._build()
        pass

    def _build(self):
        """Internal construction function.
        """
        pass

    def evaluate_batch(self, inputs, expected_outputs, test, sess):
        """
        Returns:
            a loss
        """
        pass

    def step(self, state, reward, episode, step, test, sess):
        """Computes the action for the next step.

        Args:
            state:   The current state the environment is in.
            reward:  The reward of the last action, i.e. from the last
                     transition. Given as a float value.
            episode: The current episode counter. Integer.
            step:    The current step counter. Integer.
            test:    States if this is a training step or a test step. Training
                     steps usually have exploration added while test steps do
                     not. Can be true or false.
            sess:    A tensorflow session.

        Returns:
            action: A numpy array of the next action to take.
        """
        pass

    def end_episode(self, final_state, reward, test, sess):
        """ Is called when the last state of the episode has been reached.

        Args:
            final_state: The final state of the episode.
            reward:  The reward recived by the last action
            test:    States if this is a training step or a test episode.
                     Training steps usually have exploration added while
                     test steps do not. Can be true or false.
            sess:    A tensorflow session.
        """
        pass

    def start_episode(self, start_state, test, sess):
        """ Called at the beginning of each episode. Can be
        used to initialize the agent at the beginning of each episode.
        The start_state will also be used for the first call of the
        step-function of this class.

        Args:
            start_state: The start_state of the episode.
            test:        States if this is a training step or a test step.
                         Training steps usually have exploration added while
                         test steps do not. Can be true or false.
            sess:        A tensorflow session.
        """
        pass

    def initialize(self, sess):
        """ Called exactly once when the agent is set up before the very first
        episode. This function does not get called again when the graph is be
        restored from a checkpoint and returns to an episode which is not the
        very first one.
        Can be used to initialize the agent with parameters for the very first
        episode.

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
