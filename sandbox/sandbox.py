import timeit

import numpy as np

np.random.seed(42)

SIZE = 4
NUM_NEURONS = 3

inputs = np.random.randint(low=-5, high=6, size=SIZE)


class Layer:
    """
    This class is designed to represent a layer of neurons in a neural network.
    """

    def __init__(
        self, inputs: np.ndarray, weights: np.ndarray = None, bias: int = None
    ):
        self.inputs = inputs
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(*inputs.shape)
        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.random.randint(low=-5, high=6, size=1)

    def get_output(self):
        """
        The function calculates the output by taking the sum of the element-wise
        multiplication of inputs and weights, and adding the bias.
        """
        self.output = np.dot(self.inputs, self.weights) + self.bias


inputs = np.asarray([1, 2, 3, 2.5])
weights = np.asarray(
    [
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ],
)
biases = np.asarray([2, 3, 0.5])


l0_neurons = [Layer(inputs=inputs, weights=w, bias=b) for w, b in zip(weights, biases)]
l0_outputs = []

for i, neuron in enumerate(l0_neurons):
    neuron.get_output()
    print(f"{i}: {neuron.inputs = }")
    print(f"{i}: {neuron.weights = }")
    print(f"{i}: {neuron.bias = }")
    l0_outputs.append(neuron.output)
    print(f"{i}: {neuron.output = }")

print(f"{l0_outputs = }")

l1_neurons = [Layer(inputs=np.asarray(l0_outputs)) for _ in range(len(l0_outputs))]
l1_outputs = []

for i, neuron in enumerate(l1_neurons):
    neuron.get_output()
    print(f"{i}: {neuron.inputs = }")
    print(f"{i}: {neuron.weights = }")
    print(f"{i}: {neuron.bias = }")
    l1_outputs.append(neuron.output)
    print(f"{i}: {neuron.output = }")

print(f"{l1_outputs = }")


def time_neuron_output():
    neuron_outputs = []
    for weight, bias in zip(weights, biases):
        neuron = Layer(inputs, weight, bias)
        output = neuron.get_output()
        neuron_outputs.append(output)
    return neuron_outputs


execution_time = timeit.timeit("time_neuron_output()", globals=globals(), number=10000)
print(f"Execution time over 1000 runs: {execution_time} seconds")
