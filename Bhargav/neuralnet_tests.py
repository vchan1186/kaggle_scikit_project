import numpy as np
import neuralnet as nn

nnet = nn.nnet(3,2)
nnet.initialize_weights()
W1 = nnet.W_in2hid
W2 = nnet.W_hid2out
X = np.array([[1,2,3],[4,5,6],[7,8,9]])
y = np.array([[0,1],[1,0],[1,0]]).T
nnet.train(X,y)



