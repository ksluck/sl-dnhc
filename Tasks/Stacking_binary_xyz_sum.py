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
        self._training_boxes = np.array([[12,6,9,10,5,5,11,12,11,
                                        15,10,15,13,10,2,14,7,15,
                                        13,13,10,9,13,1,13,4,11,5,
                                        3,1,14,6,2,11,12,9,2,10,5,
                                        11,8,5,11,3,7,13,4,15,8,5,
                                        8,3,13,15,9,10,11,8,6,11,
                                        15,9,1,10,14,9,11,14,15,12,
                                        2,6,1,6,7,10,15,5,14,11,15,
                                        15,14,8,11,5,14,7,13,7,11,
                                        11,8,2,8,11,6,10,11,11,15,
                                        11,15,6,10,4,8,7,14,9,3,13,
                                        12,14,12,4,7,9,11,1,8,2,4,
                                        1,13,9,6,1,1,14,9,2,10,14,
                                        10,11,11,8,4,7,13,15,6,3,6,
                                        1,1,4,11,2,10,13,4,3,4,5,12
                                        ,3,11,4,5,8,6,12,10,2,1,11,
                                        6,14,8,11,5,6,2,14,5,10,8,
                                        11,3,9,3,7,14,4,11,8,11,14
                                        ,4,13,8,13,10,11,13,15,5,3]
                                        ,[11,8,11,12,12,14,7,3,11,
                                        1,12,13,3,4,3,10,4,4,9,4,5,
                                        12,4,6,8,14,14,14,8,12,14,2,
                                        11,10,14,14,5,10,5,11,9,11,
                                        14,5,3,15,15,7,2,10,14,4,5,
                                        10,5,8,12,10,10,9,12,14,1,
                                        10,4,10,14,8,7,13,8,11,10,
                                        5,8,14,7,3,2,9,11,3,4,2,9,
                                        7,9,4,15,7,8,13,2,5,2,15,2,
                                        7,11,12,13,2,6,9,9,1,9,8,2,
                                        5,6,9,5,11,5,2,10,3,11,5,6,
                                        12,8,3,15,6,15,12,12,6,11,
                                        11,7,9,14,2,13,12,10,9,14,
                                        5,12,13,7,6,9,1,10,1,4,8,
                                        13,13,3,6,9,1,12,8,8,3,12,
                                        6,7,12,13,15,9,4,12,5,12,
                                        9,14,7,12,10,9,5,1,13,5,7
                                        ,14,9,4,9,14,9,7,2,7,4,7,
                                        5,15,1,1,15],
                                         [5,15,7,2,9,6,15,11,2,9,
                                         10,12,15,13,10,7,2,3,5,4
                                         ,3,15,3,11,9,2,11,9,7,15
                                         ,2,6,3,2,12,11,9,9,10,3,
                                         5,1,6,9,8,3,9,12,1,1,2,
                                         12,2,3,11,6,10,8,5,14,9,
                                         14,15,4,10,14,14,9,15,13,
                                         10,9,2,11,1,7,7,15,5,5,8,
                                         4,4,6,8,13,13,9,4,13,6,10,
                                         11,3,11,2,14,2,3,3,8,6,6
                                         ,6,2,5,8,6,1,10,15,7,3,7,
                                         7,3,13,7,13,10,1,9,15,10,
                                         8,9,15,15,7,14,8,1,12,14,
                                         6,8,4,8,12,13,6,12,14,10,
                                         11,9,5,4,6,7,13,8,10,15,
                                         11,12,14,9,14,4,5,14,6,11
                                         ,5,9,9,4,15,12,6,1,14,5,
                                         11,4,8,11,1,3,9,3,9,1,10,
                                         7,7,8,4,15,8,8,10,7,3,7,
                                         8,14,8,7]], dtype=np.uint8)
        self._test_boxes = np.array([    [1,7,6,1,2],
                                         [8,4,9,2,1],
                                         [2,5,8,9,12]], dtype=np.uint8)

        self._training_boxes_binary = np.zeros((3*4, self._training_boxes.shape[1]), dtype=np.float32)
        for i in range(0, self._training_boxes_binary.shape[1]):
            self._training_boxes_binary[:,i] = self._to_binary(self._training_boxes[:,i], 4)

        self._test_boxes_binary = np.zeros((3*4, self._test_boxes.shape[1]), dtype=np.float32)
        for i in range(0, self._test_boxes_binary.shape[1]):
            self._test_boxes_binary[:,i] = self._to_binary(self._test_boxes[:,i], 4)

        self._training_indicies = list(range(0,200))
        self._test_indicies = list(range(0,5))
        self._number_of_boxes = 4
        self._stacking_choices = [1., 2.,3.]#[1.0,2.0,3.0]

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
        inputs = np.zeros(shape=(nmbr_episodes, 2*self._number_of_boxes, 4*3+3))
        expected_outputs = np.zeros(shape=(nmbr_episodes, 2* self._number_of_boxes, 6))

        for ep in range(0,nmbr_episodes):
            box_choices = np.random.choice(self._training_indicies, size=(self._number_of_boxes,), replace=False)
            stacking_choice = np.random.choice(self._stacking_choices, size=(1,), replace=False)
            order_choice = np.random.permutation(range(0,self._number_of_boxes))
            #order_choice = [0,1,2,3]
            boxes_output_order = self._training_boxes[:,box_choices[order_choice[:]]]

            inputs[ep,0:self._number_of_boxes,0:12] = np.transpose(self._training_boxes_binary[:,box_choices])
            inputs[ep,self._number_of_boxes:,12] = np.ones(shape=(4,)) * stacking_choice
            #inputs[ep,self._number_of_boxes:,5] = np.array(order_choice)
            inputs[ep,self._number_of_boxes,13] = order_choice[0]
            inputs[ep,self._number_of_boxes+1,13] = order_choice[1]
            inputs[ep,self._number_of_boxes+2,13] = order_choice[2]
            inputs[ep,self._number_of_boxes+3,13] = order_choice[3]
            inputs[ep,self._number_of_boxes+1:,14] = np.ones(shape=(3,))


            expected_outputs[ep, self._number_of_boxes + 1,:] = self._to_binary_scalar(boxes_output_order[int(stacking_choice)-1, 0], 6)

            expected_outputs[ep, self._number_of_boxes + 2,:] = self._to_binary_scalar(np.sum(boxes_output_order[int(stacking_choice)-1, 0:2]), 6)

            expected_outputs[ep, self._number_of_boxes + 3,:] = self._to_binary_scalar(np.sum(boxes_output_order[int(stacking_choice)-1, 0:3]), 6)


        return inputs, expected_outputs



    def sample_from_test_set(self, nmbr_episodes):
        inputs = np.zeros(shape=(nmbr_episodes, 2 * self._number_of_boxes, 4*3+3))
        expected_outputs = np.zeros(shape=(nmbr_episodes, 2 * self._number_of_boxes, 6))

        for ep in range(0,nmbr_episodes):
            box_choices = np.random.choice(self._test_indicies, size=(self._number_of_boxes,), replace=False)
            stacking_choice = np.random.choice(self._stacking_choices, size=(1,), replace=False)
            order_choice = np.random.permutation(range(0,self._number_of_boxes))
            #order_choice = [0,1,2,3]
            boxes_output_order = self._test_boxes[:,box_choices[order_choice[:]]]

            inputs[ep,0:self._number_of_boxes,0:12] = np.transpose(self._test_boxes_binary[:,box_choices])
            inputs[ep,self._number_of_boxes:,12] = np.ones(shape=(4,)) * stacking_choice
            #inputs[ep,self._number_of_boxes:,5] = np.array(order_choice)
            inputs[ep,self._number_of_boxes,13] = order_choice[0]
            inputs[ep,self._number_of_boxes+1,13] = order_choice[1]
            inputs[ep,self._number_of_boxes+2,13] = order_choice[2]
            inputs[ep,self._number_of_boxes+3,13] = order_choice[3]
            inputs[ep,self._number_of_boxes+1:,14] = np.ones(shape=(3,))


            expected_outputs[ep, self._number_of_boxes + 1,:] = self._to_binary_scalar(boxes_output_order[int(stacking_choice)-1, 0], 6)

            expected_outputs[ep, self._number_of_boxes + 2,:] = self._to_binary_scalar(np.sum(boxes_output_order[int(stacking_choice)-1, 0:2]), 6)

            expected_outputs[ep, self._number_of_boxes + 3,:] = self._to_binary_scalar(np.sum(boxes_output_order[int(stacking_choice)-1, 0:3]), 6)

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
        return (4*3+3,)

    def get_action_size(self):
        """Returns the size of the actions.

        Returns:
            Tuple which contains the dimenions of the action.
        """
        return (6,)

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
