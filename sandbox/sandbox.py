from typing import List

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

    def forward(self, inputs: np.ndarray[float]):
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


class Relu:
    """The `Relu` class defines a method to apply the ReLU activation function to an
    input array element-wise."""

    def forward(self, inputs: np.ndarray[float]) -> np.ndarray[float]:
        """
        The function forward applies the ReLU activation function to the input
        array.

        Args:
          inputs (np.ndarray[float]): The `inputs` parameter is expected to be a
        NumPy array containing floating-point numbers. The `forward` method takes
        these inputs and applies the ReLU (Rectified Linear Unit) activation
        function element-wise, returning a list of floats where each element is the
        result of applying the ReLU function to

        Returns:
          A list of floats where each element is the maximum of 0 and the
        corresponding element in the input array.
        """
        self.output = np.maximum(0, inputs)


class Sigmoid:
    """The `Sigmoid` class defines a method to calculate the sigmoid activation
    function for an input array using the formula 1 / (1 + exp(-inputs))."""

    def forward(self, inputs: np.ndarray[float]) -> np.ndarray[float]:
        """
        The function calculates the sigmoid activation function for the input array
        using the formula 1 / (1 + exp(-inputs)).

        Args:
          inputs (np.ndarray[float]): The `inputs` parameter is expected to be a
        NumPy array containing floating-point values. The function calculates the
        sigmoid function for each element in the input array and returns a list of
        the results.

        Returns:
          the result of applying the sigmoid function to the input array.
        """
        self.output = 1 / (1 + np.exp(-inputs))


class SoftMax:
    """The SoftMax class defines a forward method that calculates the softmax
    activation for input values using NumPy arrays."""

    def forward(self, inputs: np.ndarray[float]) -> np.ndarray[float]:
        """
        The forward function calculates the softmax activation for the input values.

        Args:
          inputs (np.ndarray[float]): The `inputs` parameter in the `forward` method
        is expected to be a NumPy array of floating-point numbers. The method
        calculates the softmax function for the input values provided.
        """
        values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = values / np.sum(values, axis=1, keepdims=True)


def predict(inputs: np.ndarray[float]) -> np.ndarray[float]:
    """
    The function `predict` takes an array of floats as input and returns an array of
    indices corresponding to the maximum values along each row.

    Args:
      inputs (np.ndarray[float]): The `inputs` parameter is expected to be a NumPy
    array of floating-point numbers. The function `predict` takes this array as
    input and returns another NumPy array of floating-point numbers, which are the
    indices of the maximum values along the second axis of the input array.

    Returns:
      the index of the maximum value along axis 1 in the input array.
    """
    return np.argmax(inputs, axis=1)


relu = Relu()
sigmoid = Sigmoid()
softmax = SoftMax()

dense0 = LayerDense(n_inputs=2, n_neurons=3)
dense1 = LayerDense(n_inputs=3, n_neurons=3)
dense2 = LayerDense(n_inputs=3, n_neurons=3)

dense0.forward(inputs=X)
relu.forward(inputs=dense0.output)
dense1.forward(inputs=relu.output)
sigmoid.forward(inputs=dense1.output)
dense2.forward(inputs=relu.output)
softmax.forward(inputs=dense2.output)

print(predict(softmax.output))
