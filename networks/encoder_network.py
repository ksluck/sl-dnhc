import tensorflow as tf
import sonnet as snt
import math

class EncoderNetwork(snt.AbstractModule):

    def __init__(self, layer_sizes, name='encoder_network'):
        super(EncoderNetwork, self).__init__(name=name)
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

    def _build(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('inputs are not 2D but {}'.format(len(input.get_shape())))

        state = inputs

        for output_size in self._layer_sizes[:-1]:
            state = snt.Linear(output_size, initializers=self._get_uniform_initializers(state))(state)
            state = tf.tanh(state)

        state = snt.Linear(self._layer_sizes[-1], initializers=self._get_uniform_initializers(state), use_bias=False)(state)

        return state