import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# defining some constants
num_classes=3
batch_size =32
l0_node=64
l1_node=32

# opening file and some global constants
txtfile  = open('./seeds_dataset.txt')
rows=0
columns=0
train_row=0
test_row=0
epochs=200
learning_rate=0.01

# preprocessing function
def preprocess(txtfile):
	dataset = []
	for line in txtfile:
		curr_line = map(float,line.split())
		curr_list = list(curr_line)
		dataset.append(curr_list)
	dataset = np.asarray(dataset)
	rows = dataset.shape[0]
	columns = dataset.shape[1]
	mean = np.mean(dataset, axis =0)
	dev = np.std(dataset,axis = 0)
	mean[columns-1]=0
	dev[columns - 1] = 1
	dataset=(dataset-mean)/dev
	classes=dataset[:,columns-1].astype(int)
	col_1 = np.zeros((rows,1), dtype = int)
	col_2 = np.zeros((rows,1), dtype = int)
	col_3 = np.zeros((rows,1), dtype = int)
	col_1[classes==1]=1
	col_2[classes==2]=1
	col_3[classes==3]=1
	dataset=np.delete(dataset,columns-1,1)
	final_dataset=np.concatenate((dataset,col_1), axis=1)
	final_dataset=np.concatenate((final_dataset,col_2), axis=1)
	final_dataset = np.concatenate((final_dataset,col_3), axis=1)
	train, test = train_test_split(final_dataset, test_size = 0.2)
	pd.DataFrame(train).to_csv("./train.csv", header = None, index= None)
	pd.DataFrame(test).to_csv("./test.csv", header = None, index= None)

	return train, test, rows, columns

# data loader function for loading in batches of 32
def data_loader(train):
	i=0
	batches=[]
	while batch_size*i < train_row:
		batches.append(train[batch_size*i:batch_size*(i+1)])
		i+=1
	return batches

# randomly initializes wieght
def weight_init(x,y):
	return (np.random.uniform(low=-1, high = 1, size =(x,y)))

# one forward pass
def forward_pass(weights, result, bias):
	return result.dot(weights) + bias

#sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#deriative of sigmoid function
def sigmoid_der(x):
	sig = sigmoid(x)
	return sig*(1-sig)

# soft max function
def soft_max(x):
    exp_score = np.exp(x);
    return exp_score/np.sum(exp_score, axis = 1, keepdims = True)

# relu function
def relu(x):
	temp=x
	temp[temp<0]=0
	return temp

# deriative of relu function
def relu_der(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x;

# predict function for 1a
def predict1(train,feature, classify, wl0, wl1, biasl0, biasl1):
	sig1 = forward_pass(wl0, feature, biasl0)
	res1 = sigmoid(sig1)
	sig2 = forward_pass(wl1, res1, biasl1)
	return soft_max(sig2)	

# accuracy function for 1a
def accuracy1(train,feature, classify, wl0, wl1, biasl0, biasl1):
	feature = train[:, 0:7]
	clasify = train[:, 7:]
	res = predict1(train,feature, clasify, wl0, wl1, biasl0, biasl1)
	for i in res :
		maxVal= np.max(i)
		i[i!=maxVal]=0
		i[i==maxVal]=1
	count = 0
	for i in range(res.shape[0]):
		if((res[i]==clasify[i]).all()):
			count+=1
        
	return count*100/res.shape[0];

# predict function for 1b
def predict2(train, feature,classify,wl0,wl1,wl2, biasl01,biasl1,biasl2):
	sig0 = forward_pass(wl0,feature, biasl0)
	resultl0 = relu(sig0)
	sig1 = forward_pass(wl1,resultl0, biasl1)
	resultl1= relu(sig1)
	sig2 = forward_pass(wl2,resultl1, biasl2)
	return soft_max(sig2)

# accuracy function for 1b
def accuracy2(train, feature,classify,wl0,wl1,wl2, biasl01,biasl1,biasl2):
	feature = train[:, 0:7]
	clasify = train[:, 7:]
	res= predict2(train, feature,clasify,wl0,wl1,wl2, biasl01,biasl1,biasl2)
	for i in res:
		maxVal= np.max(i)
		i[i!=maxVal]=0
		i[i==maxVal]=1
	count = 0
	for i in range(res.shape[0]):
		if((res[i]==clasify[i]).all()):
			count+=1
	return count*100/res.shape[0]

#backward function for 1a
def backward1(resultl0, resultl1, batch):
	del3 = resultl1
	del3[batch[:,7: ]==1] -= 1
	dl1 = (resultl0.T).dot(del3)
	db1 = np.sum(del3, axis = 0, keepdims = True)

	del2 = del3.dot(wl1.T)*sigmoid_der(resultl0)
	dl0 = (batch[:,0:(columns-1)].T).dot(del2)
	db0 = np.sum(del2, axis = 0)

	return dl0,dl1,db0,db1

# training function for 1a
def training(wl0,wl1,biasl0,biasl1,train, test, batches, epochs, learning_rate):
	test_accuracy = []
	train_accuracy = []
	epoch_list = []

	for i in range(epochs):
		for batch in batches:
			#forward pass
			sig1 = forward_pass(wl0,batch[:,0:(columns-1)], biasl0)
			resultl0 = sigmoid(sig1)

			sig2 = forward_pass(wl1,resultl0, biasl1)
			resultl1= soft_max(sig2)

			denom= batch[:,(columns-1):]

			dl0,dl1,db0,db1 =backward1(resultl0, resultl1, batch)

			wl0 -= learning_rate * dl0
			wl1 -= learning_rate *  dl1
			biasl0 -= learning_rate * db0
			biasl1 -= learning_rate * db1
		if(i%10 == 0):
			# populating values for graphs
			#print('Train accuracy1 = ', accuracy1(train,batch[:,0:(columns-1)],batch[:,columns-1: ], wl0, wl1, biasl0, biasl1))
			#print('Test accuracy1 =', accuracy1(test,batch[:,0:(columns-1)],batch[:,columns-1: ], wl0, wl1, biasl0, biasl1))
			train_accuracy.append(accuracy1(train,batch[:,0:(columns-1)],batch[:,columns-1: ],wl0, wl1, biasl0, biasl1))
			test_accuracy.append(accuracy1(test,batch[:,0:(columns-1)],batch[:,columns-1: ], wl0, wl1, biasl0, biasl1))
			epoch_list.append(i);

	return test_accuracy, train_accuracy, epoch_list

#backward function for 1b
def backward2(resultl0, resultl1,resultl2, batch):
	del3 = resultl2
	del3[batch[:,7: ]==1] -= 1
	dl2 = (resultl1.T).dot(del3)
	db2 =np.sum(del3, axis = 0, keepdims = True)

	del2 = del3.dot(wl2.T)*relu_der(resultl1)
	dl1= (resultl0.T).dot(del2)
	db1 = np.sum(del2, axis = 0, keepdims = True)

	del1 = del2.dot(wl1.T)*relu_der(resultl0)
	dl0 = (batch[:,0:7].T).dot(del1)
	db0 = np.sum(del1, axis = 0, keepdims = True)

	return dl0,dl1,dl2,db0,db1,db2

# training function for 1b
def training2(wl0,wl1,wl2,biasl0,biasl1,biasl2,train, test, batches, epochs, learning_rate):
	test_accuracy = []
	train_accuracy = []
	epoch_list = []

	for i in range(epochs):
		for batch in batches:
			#forward pass
			sig0 = forward_pass(wl0,batch[:,0:(columns-1)], biasl0)
			resultl0 = relu(sig0)
			sig1 = forward_pass(wl1,resultl0, biasl1)
			resultl1= relu(sig1)
			sig2 = forward_pass(wl2,resultl1, biasl2)
			resultl2= soft_max(sig2)
			#backward prop
			dl0,dl1,dl2,db0,db1,db2 = backward2(resultl0, resultl1,resultl2, batch)

			wl0 -= learning_rate * dl0
			wl1 -= learning_rate *  dl1
			wl2 -= learning_rate * dl2
			biasl0 -= learning_rate * db0
			biasl1 -= learning_rate * db1
			biasl2 -=learning_rate * db2
		
		if(i%10 == 0):
			# populating values for graphs
			#print('Train accuracy1 = ', accuracy2(train,batch[:,0:(columns-1)],batch[:,columns-1: ], wl0, wl1,wl2, biasl0, biasl1, biasl2))
			#print('Test accuracy1 =', accuracy2(test,batch[:,0:(columns-1)],batch[:,columns-1: ], wl0, wl1,wl2, biasl0, biasl1, biasl2))
			train_accuracy.append(accuracy2(train,batch[:,0:(columns-1)],batch[:,columns-1: ],wl0, wl1,wl2, biasl0, biasl1, biasl2))
			test_accuracy.append(accuracy2(test,batch[:,0:(columns-1)],batch[:,columns-1: ], wl0, wl1,wl2, biasl0, biasl1, biasl2))
			epoch_list.append(i);

	return test_accuracy, train_accuracy, epoch_list


# main 

train,test, row, columns = preprocess(txtfile)
test_row=test.shape[0]
train_row=train.shape[0]

batches=data_loader(train)

wl0 = weight_init(columns-1,batch_size)
biasl0 = np.zeros((1,batch_size))
wl1 = weight_init(batch_size,num_classes)
biasl1 = np.zeros((1,num_classes))

test_accuracy, train_accuracy, epoch_list = training(wl0,wl1,biasl0,biasl1,train,test, batches,epochs, learning_rate)

print("Final Train accuracy = " , train_accuracy[-1])
print("Final Test accuracy = " , test_accuracy[-1])

# plots
plt.plot( epoch_list,train_accuracy, label="train accuracy")
plt.plot(epoch_list,test_accuracy, label="test accuracy" )
plt.legend(loc="upper left")
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.savefig('./costvsepoch.png')
plt.show()
print("Plot saved as costvsepoch.png")

print("beginning part 2")

wl0 = weight_init(columns-1,l0_node)
biasl0 = np.zeros((1,l0_node))
wl1 = weight_init(l0_node,l1_node)
biasl1 = np.zeros((1,l1_node))
wl2 = weight_init(l1_node,num_classes)
biasl2 = np.zeros((1,num_classes))

test_accuracy, train_accuracy, epoch_list=training2(wl0,wl1,wl2,biasl0,biasl1,biasl2,train,test,batches,epochs,learning_rate)

print("Final Train accuracy = " , train_accuracy[-1])
print("Final Test accuracy = " , test_accuracy[-1])

# plots
plt.plot( epoch_list,train_accuracy, label="train accuracy")
plt.plot(epoch_list,test_accuracy, label="test accuracy" )
plt.legend(loc="upper left")
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.savefig('./costvsepoch2.png')
plt.show()
print("Plot saved as costvsepoch2.png")