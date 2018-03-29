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
        self._training_boxes = np.array([[3,4,1,6,10,11,15,3,8,4,4,2,15,12,5,15,11,8,6,8,7,1,6,12,12,13,7,3,13,2,2,2,8,6,8,4,15,7,3,8]
                                        ,[2,9,13,7,2,7,3,8,15,15,11,8,11,8,12,4,4,4,12,11,14,5,14,9,7,12,13,1,13,11,7,14,14,8,7,13,15,7,7,11],
                                         [5,2,15,12,14,2,11,12,13,14,8,11,4,8,6,7,12,8,1,12,14,7,7,6,8,7,5,7,8,11,15,11,5,3,2,15,3,15,4,1]], dtype=np.uint8)
        self._test_boxes = np.array([    [1,7,6,1,2],
                                         [8,4,9,2,1],
                                         [2,5,8,9,12]], dtype=np.uint8)

        self._training_boxes_binary = np.zeros((3*4, self._training_boxes.shape[1]), dtype=np.float32)
        for i in range(0, self._training_boxes_binary.shape[1]):
            self._training_boxes_binary[:,i] = self._to_binary(self._training_boxes[:,i], 4)

        self._test_boxes_binary = np.zeros((3*4, self._test_boxes.shape[1]), dtype=np.float32)
        for i in range(0, self._test_boxes_binary.shape[1]):
            self._test_boxes_binary[:,i] = self._to_binary(self._test_boxes[:,i], 4)

        self._training_indicies = list(range(0,40))
        self._test_indicies = list(range(0,5))
        self._number_of_boxes = 4
        self._stacking_choices = [3.0]#[1.0,2.0,3.0]

    def _to_binary(self,ar, m):
        ar_1 = np.array(list(np.binary_repr(ar[0]).zfill(m))).astype(np.float32)
        ar_2 = np.array(list(np.binary_repr(ar[1]).zfill(m))).astype(np.float32)
        ar_3 = np.array(list(np.binary_repr(ar[2]).zfill(m))).astype(np.float32)
        return np.concatenate((ar_1, ar_2, ar_3))


    def _to_binary_scalar(self, val, m):
        val_bin = np.array(list(np.binary_repr(val).zfill(m))).astype(np.float32)
        return val_bin



    def sample_set(self, nmbr_episodes, test =False):
        if test:
            return self.sample_from_test_set(nmbr_episodes=nmbr_episodes)
        else:
            return self.sample_from_training_set(nmbr_episodes=nmbr_episodes)

    def sample_from_training_set(self, nmbr_episodes):
        inputs = np.zeros(shape=(nmbr_episodes, 2*self._number_of_boxes, 4+3))
        expected_outputs = np.zeros(shape=(nmbr_episodes, 2*self._number_of_boxes, 4))

        for ep in range(0,nmbr_episodes):
            box_choices = np.random.choice(self._training_indicies, size=(self._number_of_boxes,), replace=False)
            stacking_choice = np.random.choice(self._stacking_choices, size=(1,), replace=False)
            order_choice = np.random.permutation(range(0,self._number_of_boxes))
            #order_choice = [0,1,2,3]
            boxes_output_order = self._training_boxes[:,box_choices[order_choice[:]]]

            inputs[ep,0:self._number_of_boxes,0:4] = np.transpose(self._training_boxes_binary[8:12,box_choices])
            inputs[ep,self._number_of_boxes:,4] = np.ones(shape=(self._number_of_boxes,)) * stacking_choice
            inputs[ep,self._number_of_boxes:,5] = np.array(order_choice)
            inputs[ep,self._number_of_boxes:,6] = np.ones(shape=(self._number_of_boxes,))

            for i in range(0,self._number_of_boxes):
                expected_outputs[ep, self._number_of_boxes + i,:] = self._to_binary_scalar(boxes_output_order[int(stacking_choice)-1, i], 4)

        return inputs, expected_outputs



    def sample_from_test_set(self, nmbr_episodes):
        inputs = np.zeros(shape=(nmbr_episodes, 2*self._number_of_boxes, 4+3))
        expected_outputs = np.zeros(shape=(nmbr_episodes, 2*self._number_of_boxes, 4))

        for ep in range(0,nmbr_episodes):
            box_choices = np.random.choice(self._test_indicies, size=(self._number_of_boxes,), replace=False)
            stacking_choice = np.random.choice(self._stacking_choices, size=(1,), replace=False)
            order_choice = np.random.permutation(range(0,self._number_of_boxes))
            #order_choice = [0,1,2,3]
            boxes_output_order = self._test_boxes[:,box_choices[order_choice[:]]]

            inputs[ep,0:self._number_of_boxes,0:4] = np.transpose(self._test_boxes_binary[8:12,box_choices])
            inputs[ep,self._number_of_boxes:,4] = np.ones(shape=(self._number_of_boxes,)) * stacking_choice
            inputs[ep,self._number_of_boxes:,5] = np.array(order_choice)
            inputs[ep,self._number_of_boxes:,6] = np.ones(shape=(self._number_of_boxes,))

            for i in range(0,self._number_of_boxes):
                expected_outputs[ep, self._number_of_boxes + i,:] = self._to_binary_scalar(boxes_output_order[int(stacking_choice)-1, i], 4)

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
        return (7,)

    def get_action_size(self):
        """Returns the size of the actions.

        Returns:
            Tuple which contains the dimenions of the action.
        """
        return (4,)

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
