"""
Code written by Kevin S. Luck
2018
"""

class Logger():
    """ Basis class for loggers.
    Provides at the moment only functionality for logging the rewards.
    """
    def __init__(self, config = None):
        """ Initializes the logger.

        Args:
            config: Dictonary which contains configs
        """
        pass

    def logLoss(self, nmbr_steps, episode, loss, test=False, sess = None, agent_id=0):
        """ Function to log the episodic reward.

        Args:
          nmbr_steps: Number of steps performed this episode.
          episode: number of the current episode.
          loss: loss achieved throughout the episode.
          sess: A tensorflow session
          agent_id: ID of agent, not used currently.
        """
        pass
