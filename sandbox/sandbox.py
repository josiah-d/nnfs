import numpy as np

np.random.seed(42)

SIZE = 4
NUM_NEURONS = 3

inputs = np.random.randint(low=-5, high=6, size=SIZE)


class Neuron:
    """
    This class is designed to represent a single neuron in a neural network.
    """

    def __init__(self, inputs):
        """
        The function initializes weights and bias with random values for a neural
        network.

        Args:
          inputs: The `inputs` parameter in the `__init__` method of your code
        snippet is used to initialize the instance variable `self.inputs`. This
        parameter is expected to be a numpy array representing the input data for
        your model. The `weights` variable is then initialized with random values of
        the same shape
        """
        self.inputs = inputs
        self.weights = np.random.rand(*inputs.shape)
        self.bias = np.random.randint(low=-5, high=6, size=1)

    def get_output(self):
        """
        The function calculates the output by taking the sum of the element-wise
        multiplication of inputs and weights, and adding the bias.
        """
        self.output = np.sum(inputs * self.weights) + self.bias


l0_neurons = [Neuron(inputs=inputs) for _ in range(NUM_NEURONS)]
l0_outputs = []

for i, neuron in enumerate(l0_neurons):
    neuron.get_output()
    print(f"{i}: {neuron.inputs = }")
    print(f"{i}: {neuron.weights = }")
    print(f"{i}: {neuron.bias = }")
    l0_outputs.append(neuron.output)
    print(f"{i}: {neuron.output = }")

l1_neurons = [Neuron(inputs=np.asarray(l0_outputs)) for _ in range(NUM_NEURONS)]
l1_outputs = []

for i, neuron in enumerate(l1_neurons):
    neuron.get_output()
    print(f"{i}: {neuron.inputs = }")
    print(f"{i}: {neuron.weights = }")
    print(f"{i}: {neuron.bias = }")
    l1_outputs.append(neuron.output)
    print(f"{i}: {neuron.output = }")
