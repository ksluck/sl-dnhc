"""
Code written by Kevin S. Luck
2018
"""
import SL.Task as Task
import numpy as np

class Stacking(Task.Task):
    """ This class is the base class for each environment (wrapper).
    """
    def __init__(self, config, render):
        """ Sets up class with config and renderer.

        Args:
            config: A dictonary which contains config-options.
            render: Boolean whihc renders environment if true.
        """
        self._training_boxes = np.array([[64,86,50,19,124,34,13,24,83,13,113,31,54,53,133,87,67,108,28,116,45,34,21,35,5,15,149,44,21,104,26,114,139,92,146]
                                        ,[91,144,125,42,37,86,80,127,2,140,48,21,37,101,86,88,57,104,142,66,127,98,42,5,63,48,50,58,43,24,22,57,128,25,94],
                                         [39,121,69,62,20,9,95,125,34,48,51,33,141,101,99,58,78,13,140,87,57,80,30,106,145,129,32,110,119,9,61,89,51,70,91]], dtype=np.float64) / 10.0
        self._test_boxes = np.array([    [17,8,94,25,127],
                                         [149,115,58,96,56],
                                         [131,64,45,134,52]], dtype=np.float64) / 10.0
        self._training_indicies = list(range(0,9))
        self._test_indicies = list(range(0,5))
        self._number_of_boxes = 4
        self._stacking_choices = [-1.,0.,1.]

    def sample_set(self, nmbr_episodes, test =False):
        if test:
            return self.sample_from_test_set(nmbr_episodes=nmbr_episodes)
        else:
            return self.sample_from_training_set(nmbr_episodes=nmbr_episodes)

    def sample_from_training_set(self, nmbr_episodes):
        inputs = np.zeros(shape=(nmbr_episodes, 2*self._number_of_boxes, 6))
        expected_outputs = np.zeros(shape=(nmbr_episodes, 2*self._number_of_boxes, 3))

        for ep in range(0,nmbr_episodes):
            box_choices = np.random.choice(self._training_indicies, size=(self._number_of_boxes,), replace=False)
            stacking_choice = np.random.choice(self._stacking_choices, size=(1,), replace=False)
            order_choice = np.random.permutation(range(0,self._number_of_boxes))
            boxes_output_order = self._training_boxes[:,box_choices[order_choice[:]]]

            inputs[ep,0:self._number_of_boxes,0:3] = np.transpose(self._training_boxes[:,box_choices])
            inputs[ep,self._number_of_boxes:,3] = np.ones(shape=(self._number_of_boxes,))
            inputs[ep,self._number_of_boxes:,4] = np.ones(shape=(self._number_of_boxes,)) * stacking_choice
            inputs[ep,self._number_of_boxes:,5] = np.array(order_choice)

            for i in range(1,self._number_of_boxes):
                expected_outputs[ep, self._number_of_boxes + i,int(stacking_choice) + 1] = np.sum( boxes_output_order[int(stacking_choice)+1, range(0,i)])

        return inputs, expected_outputs



    def sample_from_test_set(self, nmbr_episodes):
        inputs = np.zeros(shape=(nmbr_episodes, 2*self._number_of_boxes, 6))
        expected_outputs = np.zeros(shape=(nmbr_episodes, 2*self._number_of_boxes, 3))

        for ep in range(0,nmbr_episodes):
            box_choices = np.random.choice(self._test_indicies, size=(self._number_of_boxes,), replace=False)
            stacking_choice = np.random.choice(self._stacking_choices, size=(1,), replace=False)
            order_choice = np.random.permutation(range(0,self._number_of_boxes))
            boxes_output_order = self._test_boxes[:,box_choices[order_choice[:]]]

            inputs[ep,0:self._number_of_boxes,0:3] = np.transpose(self._test_boxes[:,box_choices])
            inputs[ep,self._number_of_boxes:,3] = np.ones(shape=(self._number_of_boxes,))
            inputs[ep,self._number_of_boxes:,4] = np.ones(shape=(self._number_of_boxes,)) * stacking_choice
            inputs[ep,self._number_of_boxes:,5] = np.array(order_choice)

            for i in range(1,self._number_of_boxes):
                expected_outputs[ep, self._number_of_boxes + i,int(stacking_choice) + 1] = np.sum( boxes_output_order[int(stacking_choice)+1, range(0,i)])

        return inputs, expected_outputs

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
        return (6,)

    def get_action_size(self):
        """Returns the size of the actions.

        Returns:
            Tuple which contains the dimenions of the action.
        """
        return (3,)

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
