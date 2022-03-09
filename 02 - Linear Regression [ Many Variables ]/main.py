import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def computeCost(X, y, theta):
  z = np.power(((X * theta.T) - y), 2)
  return np.sum(z) / (2 * len(X))
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

path = 'data.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

# data:
print(data.head(10))
print("#" * 50)

# description data:
print(data.describe())
print("#" * 50)

# rescaling data
data = (data - data.mean()) / data.std()

# data after rescaling
print(data.head(10))
print("#" * 50)

data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.matrix(np.array([0, 0, 0]))

alpha = 0.1
iters = 100

g, cost = gradientDescent(X, y, theta, alpha, iters)

thisCost = computeCost(X, y, g)

x1 = np.linspace(data.Size.min(), data.Size.max(), 100)
print(x1)

print("#" * 50)

f = g[0, 0] + (g[0, 1] * x1)
print(f)

plt.plot(x1, f, 'r')
plt.scatter(data.Size, data.Price)
plt.grid()
plt.show()

# errors graph

plt.plot(np.arange(iters), cost)
plt.show()
