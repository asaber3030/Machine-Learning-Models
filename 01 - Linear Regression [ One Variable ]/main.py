import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'data.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

print('data = \n', data.head(10))
print("-" * 100)
print('data.describe = \n', data.describe())
print("-" * 100)

plt.scatter(data['Population'], data['Profit'])

data.insert(0, 'Ones', 1)
print('new data = \n', data.head(10))
print("-" * 100)

cols = data.shape[1]
X = data.iloc[:,0:cols - 1]
y = data.iloc[:,cols-1:cols]
print('x = \n', X.head(10))
print('y = \n', y.head(10))
print("-" * 100)

X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.matrix(np.array([0, 0]))

print('x = \n', X)
print('x shape = \n', X.shape)

print('theta = \n', theta)
print('theta shape = \n', theta.shape)

print('y = \n', y)
print('y shape = \n', y.shape)

print("-" * 100)

def computeCost(X, y, theta):
  z = np.power(((X * theta.T) - y), 2)
  return np.sum(z) / (2 * len(X))

print('computeCostFunction = \n', computeCost(X, y, theta))

print("-" * 100)

def gradientDescent(X, y, theta, alpha, iters):
  temp = np.matrix(np.zeros(theta.shape))
  params = int(theta.ravel().shape[1])
  cost = np.zeros(iters)

  for i in range(iters):
    error = (X * theta.T) - y # (h(x) - y)
    for j in range(params):
      term = np.multiply(error, X[:,j])
      temp[0,j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

    theta = temp
    cost[i] = computeCost(X, y, theta)

  return theta, cost

alpha = 0.01
iters = 1000
g, cost = gradientDescent(X, y, theta, alpha, iters)
print('g = ', g)
print('cost = ', cost[0:50])
print('computeCostFunction = \n', computeCost(X, y, theta))

print("-" * 100)

# Best Fit line
x = np.linspace(data.Population.min(), data.Population.max(), 100)
print('x = \n', x)
print('g = \n', g)

print("-" * 100)

f = g[0, 0] + (g[0, 1] * x)
print('f = \n', f)

plt.plot(x, f, 'r')
plt.grid()
plt.show()

