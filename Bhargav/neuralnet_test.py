import numpy as np
import neuralnet as nn
import unittest

class testNN(unittest.TestCase):

	def setUp(self):
		self.inputDim = 4
		self.numHidden = 3
		self.outputDim = 1
		self.data = np.array([
			[0.1,0.2,0.9,0.7],
			[0,11,0.21,0.91,0.71],
			[0.12,0.22,0.92,0.72],
			[0.9,0.7,0.1,0.2],
			[0.91,0.71,0.11,0.21],
			[0.92,0.72,0.12,0.22]
			])
		self.targets = np.array([0],[0],[0],[1],[1],[1])
		self.nnet = nn.nnet(inputDim,numHidden,outputDim)

	def fprop_test(self):
		self.nnet.initialize_weights()

def main():
	unittest.main()

if __name__ == '__main__':
	main()


