import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--and", type = int, default = 0)
ap.add_argument("-o", "--or", type = int, default = 0)
ap.add_argument("-xr", "--xor", type = int, default = 0)
args = vars(ap.parse_args())

an = args["and"]
on = args["or"]
xrn = args["xor"]

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
a = np.array([[0], [0], [0], [1]])
o = np.array([[0], [1], [1], [1]])
xr = np.array([[0], [1], [1], [0]])

class Perceptron:
	def __init__(self, N, alpha=0.1):
		self.W = np.random.randn(N + 1) / np.sqrt(N)
		self.alpha = alpha

	def step(self, x):
		return 1 if x > 0 else 0

	def fit(self, X, y, epochs=10):
		X = np.c_[X, np.ones((X.shape[0]))]
		
		for epoch in np.arange(0, epochs):
			for (x, target) in zip(X, y):
			
				p = self.step(np.dot(x, self.W))

				if p != target:
					error = p - target
					self.W += -self.alpha * error * x

	def predict(self, X, addBias=True):
		X = np.atleast_2d(X)
		
		if addBias:
			X = np.c_[X, np.ones((X.shape[0]))]
			
		return self.step(np.dot(X, self.W))

if an == 1:
	p = Perceptron(X.shape[1], alpha=0.1)
	p.fit(X, a, epochs=20)
	
	for (x, target) in zip(X, a):
		pred = p.predict(x)
		print("[INFO] [AND] data={}, ground-truth={}, pred={}".format(x, target[0], pred))
if on == 1:
	p = Perceptron(X.shape[1], alpha=0.1)
	p.fit(X, o, epochs=20)
	
	for (x, target) in zip(X, o):
		pred = p.predict(x)
		print("[INFO] [OR] data={}, ground-truth={}, pred={}".format(x, target[0], pred))
if xrn == 1:
	p = Perceptron(X.shape[1], alpha=0.1)
	p.fit(X, xr, epochs=20)
	
	for (x, target) in zip(X, xr):
		pred = p.predict(x)
		print("[INFO] [XOR] data={}, ground-truth={}, pred={}".format(x, target[0], pred))
