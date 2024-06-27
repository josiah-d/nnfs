import numpy as np
from nnfs.datasets import spiral_data

np.random.seed(42)

SIZE = 4
NUM_NEURONS = 3

X, y = spiral_data(samples=100, classes=3)


class LayerDense:
    """
    This class is designed to represent a LayerDense of neurons in a neural network.
    """

    def __init__(self, n_inputs: int, n_neurons: int):
        """
        This function initializes weights and biases for a neural network with specified
        number of inputs and neurons.

        Args:
          n_inputs (int): The `n_inputs` parameter represents the number of input
        features in a neural network layer. It is the number of nodes in the previous
        layer or the number of input values provided to the current layer.
          n_neurons (int): The `n_neurons` parameter represents the number of neurons in
        the neural network layer for which you are initializing the weights and biases.
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray):
        """
        The forward function calculates the output by multiplying inputs with
        weights, summing the results, and adding biases.

        Args:
          inputs (np.ndarray): The `inputs` parameter in the `forward` function
        represents the input data that is fed into the neural network layer. It is
        expected to be a NumPy array containing the input values for the layer. The
        function calculates the output of the layer by performing a dot product
        between the input data and the
        """
        self.output = np.dot(inputs, self.weights) + self.biases


dense0 = LayerDense(n_inputs=2, n_neurons=3)
dense1 = LayerDense(n_inputs=3, n_neurons=4)
dense2 = LayerDense(n_inputs=4, n_neurons=2)

dense0.forward(inputs=X)
dense1.forward(inputs=dense0.output)
dense2.forward(inputs=dense1.output)

print(dense2.output.shape)
print(dense2.output)
