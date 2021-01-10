import numpy as np
from sklearn.neural_network import MLPClassifier

#function to ignore warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#some global variables
learn_rate = 0.01
epochs = 200
batch_s = 32
nodes_1=32
nodes_2=64

# function to get accuracy
def accuracy(predicted, actual):
	count = 0
	rows= predicted.shape[0]
	for i in range(rows):
		if((predicted[i]==actual[i]).all()):
			count+=1
	return 100*count/rows

train= np.genfromtxt('./train.csv', delimiter = ',')
test = np.genfromtxt('./test.csv', delimiter = ',')

x_train = train[:, :7]
y_train = train[:, 7:]

x_test = test[:, :7]
y_test = test[:, 7:]

# MLP classifiers
classifierf1 = MLPClassifier(solver = 'sgd', activation = 'logistic',batch_size = batch_s,
                    hidden_layer_sizes = (nodes_1), random_state =1, learning_rate_init = learn_rate, learning_rate = 'constant', max_iter = epochs)

classifierf2 = MLPClassifier(solver = 'sgd', activation = 'relu',batch_size = batch_s,
                    hidden_layer_sizes = (nodes_2, nodes_1), random_state =1, learning_rate_init = learn_rate, learning_rate = 'constant', max_iter = epochs)

classifierf1.fit(x_train, y_train)
classifierf2.fit(x_train,y_train)

predicted_train1 = classifierf1.predict(x_train)
predicted_test1 = classifierf1.predict(x_test)

predicted_train2 = classifierf2.predict(x_train)
predicted_test2 = classifierf2.predict(x_test)


#results
print("train accuracy for 1a(scikit) = ", accuracy(predicted_train1,y_train))
print("test accuracy for 1a(scikit) = ", accuracy(predicted_test1,y_test))
print("train accuracy for 1b(scikit) = ", accuracy(predicted_train2,y_train))
print("test accuracy for 1b(scikit) = ", accuracy(predicted_test2,y_test))

