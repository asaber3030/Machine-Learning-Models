import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

predict = np.array(([4, 8]), dtype=float)

# Scaling by divide every number on the max number in (n-th)-col
X = X / np.amax(X, axis=0)
predict = predict / np.amax(predict, axis=0)
y = y / 100

class NeuralNetwork(object):

  def __init__(self):

    self.inputSize = 2 # 2 features
    self.outputSize = 1
    self.hiddenSize = 3 # 3 neurons in hidden layer

    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (2x3)
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # 3x1

  def sigmoid(self, z):
    return (1 / (1 + np.exp(-z)))

  def derivative_sigmoid(self, x):
    return x * (1 - x)

  def forward(self, X):
    self.z = np.dot(X, self.W1)
    self.z2 = self.sigmoid(self.z)
    self.z3 = np.dot(self.z2, self.W2)
    return self.sigmoid(self.z3)

  def backward(self, X, y, o):
    self.o_error = y - o
    self.o_delta = self.o_error * self.derivative_sigmoid(o)

    self.z2_error = self.o_delta.dot(self.W2.T)
    self.z2_delta = self.z2_error * self.derivative_sigmoid(self.z2)

    self.W1 += X.T.dot(self.z2_delta)
    self.W2 += self.z2.T.dot(self.o_delta)

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def predict(self):
    print("My Inputs predictions: \n", str(self.forward(predict)))

  def save(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")


nn = NeuralNetwork()
iters = 100000

for i in range(iters):
  print("#" * 100)
  print("#" + str(i))
  print("Predictions \n" + str(nn.forward(X)))

  print("Loss " + str(np.mean(np.square(y - nn.forward(X)))))
  nn.train(X, y)

print("#" * 100)

nn.save()
nn.predict()
