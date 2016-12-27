# visualizing logistic regression vs neural network
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.visualize_util import plot

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]  # we only take the first two features.
Y = iris.target
#
# data = datasets.load_breast_cancer()
# X = data.data[:, [0, 2]]
# Y = data.target

h = .02  # step size in the mesh

model = Sequential()
model.add(Dense(12, input_dim=2, init='uniform', activation='sigmoid'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=15)

plot(model, to_file='model.png')


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
