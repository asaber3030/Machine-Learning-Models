import numpy as np

neurons = 4

def sigmoid(z):
  return (1 / (1 + np.exp(-z)))

def derivative_sigmoid(x):
  return x * ( 1 - x )

class NeuralNetwork:
  def __init__(self, X, y):
    self.input = X
    print("Inputs \n", self.input)
    print("#" * 100)

    self.weights1 = np.random.rand(self.input.shape[1], 4)
    print("Weights1 \n", self.weights1)
    print("#" * 100)

    self.weights2 = np.random.rand(neurons, 1)
    print("Weights2 \n", self.weights2)
    print("#" * 100)

    self.y = y
    print("y \n", y)
    print("#" * 100)

    self.output = np.zeros(self.y.shape)
    print("output \n", self.output)
    print("#" * 100)

  def forwardPropagation(self):
    self.layer1 = sigmoid(np.dot(self.input, self.weights1))
    # print("Layer1 \n", self.layer1)
    # print("#" * 100)

    self.output = sigmoid(np.dot(self.layer1, self.weights2))
    # print("Output \n", self.output)
    # print("#" * 100)


  def backPropagation(self):
    d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * derivative_sigmoid(self.output)))
    d_weights1 = np.dot(
      self.input.T,
      (np.dot(
        2 * (self.y - self.output) * derivative_sigmoid(self.output), self.weights2.T)
      * derivative_sigmoid(self.layer1)))

    self.weights1 += d_weights1
    self.weights2 += d_weights2

X = np.array([
  [0, 0, 1],
  [0, 1, 1],
  [1, 0, 1],
  [1, 1, 1]
])

y = np.array([
  [0],
  [1],
  [1],
  [0]
]) # Real values

nn = NeuralNetwork(X, y)
iterations = 10000

for i in range(iterations):
  nn.forwardPropagation()
  nn.backPropagation()


print("Final Output \n", nn.output) # Expected values h(x)
# Output
# [
#   [0.0045573 ]
#   [0.98968118]
#   [0.99417126]
#   [0.01038239]
# ]