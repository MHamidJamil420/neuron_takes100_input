# Muhammad Hamid Jamil
# SP19-BCS-098
import numpy as np
import random


class Neuron:
    def __init__(self, no_of_inputs):
        self.bias = 0.2
        self.weights = [0.65, 0.30]
        self.learning_rate = 0.1

    def predict(self, input1, input2):
        summation = input1 * self.weights[0] + input2 * self.weights[1] + self.bias
        if summation >= 0:
            return 1
        else:
            return 0

    def train(self, training_ex, actual_label):
        predicted_label = self.predict(training_ex[0], training_ex[1])
        print("predicted_label is : ", predicted_label, "bias is : ", self.bias)
        if predicted_label != actual_label:
            error = actual_label - predicted_label
            delta_w = self.learning_rate * error
            self.weights[0] += delta_w
            self.weights[1] += delta_w
            self.bias += delta_w
            print("error is :", error, "new bias is : ", self.bias, "predicted_label is : ", predicted_label)
            self.train(training_ex, actual_label)
        return actual_label


# this model generates 100 random inputs and predefined output,
# so I want 0 when the difference between those (random) input is odd, and 1 when difference
# between those (random) input is even
for number in range(100):
    x1 = random.randint(0, 99)
    x2 = random.randint(0, 99)
    print("X1 is : ", x1, "X2 is : ", x2)
    z = x1 - x2
    training_input = np.array([x1, x2])
    neuron = Neuron(2)
    if z < 0:
        z *= -1

    if (z % 2) == 0:
        Y = 1
        print("1, dif : ", z)
    else:
        Y = 0
        print("0, dif : ", z)

    decision = neuron.train(training_input, Y)
    print("the decision is : ", decision)

# training_input = np.array([x1, x2])
# neuron = Neuron(2)
# decision = neuron.train(training_input, Y)
# print("the decision is : ", decision)
