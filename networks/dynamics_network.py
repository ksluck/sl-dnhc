import tensorflow as tf
import sonnet as snt
import math

class DynamicsNetwork(snt.AbstractModule):

    def __init__(self, layer_sizes, name = 'dynamics'):
        super(DynamicsNetwork, self).__init__(name=name)
        self._layer_sizes = layer_sizes

    def _get_uniform_initializers(self, inputs):
        stddev = 1.0/math.sqrt(inputs.get_shape()[1].value)
        def _initializer(shape, dtype=tf.float32, partition_info = None):
            del partition_info
            return tf.random_uniform(shape, -stddev, stddev, dtype)
        return {
            'w': _initializer,
            'b': _initializer,
        }

    def _build(self, state, action):
        if len(state.get_shape()) != 2:
            raise ValueError('inputs are not 2D but {}'.format(len(input.get_shape())))
        if len(action.get_shape()) != 2:
            raise ValueError('inputs are not 2D but {}'.format(len(input.get_shape())))

        state_length = state.get_shape()[1]

        state = (snt.Linear(self._layer_sizes[0], initializers=self._get_uniform_initializers(state))(state) +
                 snt.Linear(self._layer_sizes[0], initializers=self._get_uniform_initializers(action))(action))
        state = tf.tanh(state)

        for output_size in self._layer_sizes[1:]:
            state = snt.Linear(output_size, initializers=self._get_uniform_initializers(state))(state)

        state = snt.Linear(state_length, initializers=self._get_uniform_initializers(state))(state)

        return state