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
        self._training_boxes = np.array([[22,111,117,129,58,70,62,47,144,134,123,
                                            96,90,101,117,124,76,139,100,56,31,
                                            109,110,117,71,95,71,126,59,56,50,
                                            10,132,144,129,55,64,101,18,133,7,
                                            96,113,79,144,131,100,147,99,9,56,
                                            140,42,16,138,128,10,146,76,51,42,
                                            132,84,75,47,149,80,70,136,90,54,83,
                                            69,122,27,52,32,140,70,11,89,138,39,
                                            12,51,75,97,50,58,148,133,120,4,68,
                                            55,23,136,1,54,133,131,138,39,36,15,
                                            116,5,127,27,83,30,64,34,37,48,137,
                                            20,119,12,17,96,133,27,69,49,81,18,
                                            105,37,126,99,46,93,121,126,135,130,
                                            9,144,97,74,55,55,32,35,145,86,9,144,
                                            110,85,105,17,102,32,145,68,146,2,
                                            53,77,14,55,53,70,95,97,79,40,120,5,
                                            70,88,130,86,65,45,28,80,107,107,5,
                                            33,141,99,1,2,109,26,83,130,100,121
                                            ,70,14,79,78,142,137,145],
                                         [28,146,103,102,100,128,131,105,137,
                                         101,57,35,119,113,97,142,18,136,25,91,
                                         58,140,60,51,119,45,118,84,103,136,136,
                                         130,77,57,55,2,138,113,26,123,85,78,33,
                                         88,72,25,84,15,14,144,68,64,61,12,104,
                                         100,131,142,21,116,90,60,42,143,76,92,
                                         14,61,51,51,112,99,61,36,148,35,138,93,
                                         70,25,76,18,85,8,104,116,62,121,95,84,
                                         131,101,48,125,80,37,84,45,144,100,125,
                                         82,39,34,46,132,48,52,136,120,41,44,50,
                                         88,12,137,85,76,20,131,112,2,126,27,10,
                                         39,11,121,91,27,16,12,3,59,139,43,40,
                                         51,26,8,109,32,48,25,149,31,3,121,138,
                                         37,119,33,138,70,105,45,85,110,26,146,
                                         30,106,92,9,36,27,15,8,125,131,92,31,
                                         75,95,11,32,72,73,35,32,77,41,65,81,87,
                                         69,140,19,127,78,94,45,14,49,135,63,
                                         109,116,41,29],
                                         [74,10,29,6,145,47,46,90,54,119,72,78,
                                         82,101,112,32,145,145,130,9,98,10,80,
                                         132,82,114,124,86,23,62,110,23,10,144,
                                         19,134,66,76,118,99,36,38,42,135,91,
                                         144,120,39,101,44,20,135,28,6,150,23,
                                         56,135,115,137,92,27,75,21,114,108,94,
                                         72,45,58,65,138,100,25,124,45,90,95,51,
                                         143,142,118,42,1,5,54,48,109,148,11,14,
                                         37,96,86,33,22,33,108,64,84,42,120,77,
                                         146,81,52,17,48,124,67,18,80,92,116,
                                         100,53,8,128,69,51,42,44,79,98,101,33,
                                         36,73,79,130,15,4,31,14,48,34,35,124,
                                         131,89,56,89,16,13,149,106,11,128,56,
                                         18,49,127,102,116,100,65,108,101,80,24,
                                         133,82,141,105,139,41,17,44,94,5,76,
                                         106,130,28,102,27,76,29,46,106,53,125,
                                         102,148,117,52,29,63,27,27,147,16,76,
                                         11,29,105,22,89,20,16]], dtype=np.float64) / 10.0
        self._test_boxes = np.array([    [17,8,94,25,127],
                                         [149,115,58,96,56],
                                         [131,64,45,134,52]], dtype=np.float64) / 10.0
        self._training_indicies = list(range(0,self._training_boxes.shape[1]))
        self._test_indicies = list(range(0,self._test_boxes.shape[1]))
        self._number_of_boxes = 4
        self._stacking_choices = [1.0,2.0,3.0]

    def sample_set(self, nmbr_episodes, test =False):
        if test:
            return self.sample_from_test_set(nmbr_episodes=nmbr_episodes)
        else:
            return self.sample_from_training_set(nmbr_episodes=nmbr_episodes)

    def sample_from_training_set(self, nmbr_episodes):
        inputs = np.zeros(shape=(nmbr_episodes, 2*self._number_of_boxes, 6))
        expected_outputs = np.zeros(shape=(nmbr_episodes, 2*self._number_of_boxes, 1))

        for ep in range(0,nmbr_episodes):
            box_choices = np.random.choice(self._training_indicies, size=(self._number_of_boxes,), replace=False)
            stacking_choice = np.random.choice(self._stacking_choices, size=(1,), replace=False)
            order_choice = np.random.permutation(range(0,self._number_of_boxes))
            boxes_output_order = self._training_boxes[:,box_choices[order_choice[:]]]

            inputs[ep,0:self._number_of_boxes,0:3] = np.transpose(self._training_boxes[:,box_choices])
            inputs[ep,self._number_of_boxes+1:,3] = np.ones(shape=(self._number_of_boxes - 1,))
            inputs[ep,self._number_of_boxes:,4] = np.ones(shape=(self._number_of_boxes,)) * stacking_choice
            inputs[ep,self._number_of_boxes:,5] = np.array(order_choice)


            for i in range(1,self._number_of_boxes):
                expected_outputs[ep, self._number_of_boxes + i,0] = np.sum( boxes_output_order[int(stacking_choice)-1, range(0,i)])

        return inputs, expected_outputs



    def sample_from_test_set(self, nmbr_episodes):
        inputs = np.zeros(shape=(nmbr_episodes, 2*self._number_of_boxes, 6))
        expected_outputs = np.zeros(shape=(nmbr_episodes, 2*self._number_of_boxes, 1))

        for ep in range(0,nmbr_episodes):
            box_choices = np.random.choice(self._test_indicies, size=(self._number_of_boxes,), replace=False)
            stacking_choice = np.random.choice(self._stacking_choices, size=(1,), replace=False)
            order_choice = np.random.permutation(range(0,self._number_of_boxes))
            boxes_output_order = self._test_boxes[:,box_choices[order_choice[:]]]

            inputs[ep,0:self._number_of_boxes,0:3] = np.transpose(self._test_boxes[:,box_choices])
            inputs[ep,self._number_of_boxes+1:,3] = np.ones(shape=(self._number_of_boxes - 1,))
            inputs[ep,self._number_of_boxes:,4] = np.ones(shape=(self._number_of_boxes,)) * stacking_choice
            inputs[ep,self._number_of_boxes:,5] = np.array(order_choice)

            for i in range(1,self._number_of_boxes):
                expected_outputs[ep, self._number_of_boxes + i,0] = np.sum( boxes_output_order[int(stacking_choice)-1, range(0,i)])

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
        return (1,)

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
