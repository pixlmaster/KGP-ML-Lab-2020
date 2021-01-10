import pandas as pd
import numpy as np
import math
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

##### 2.1 custom logistic regression

#sigmoid function
def sig(x):
	return 1 / (1 + np.exp(-x))
# cost function
def cost(x,q,theta,m):
	hdotx=np.dot(x,np.transpose(theta))
	hx = sig(hdotx)
	loghx = np.log(hx)
	log1hx = np.log(1-hx)
	value = -q*loghx -(1-q)*log1hx
	value = (np.sum(value))/m
	return value
# function to update theta
def update(x,q,theta,m, alpha):
	hdotx=np.dot(x,np.transpose(theta))
	x_t = np.transpose(x)
	axdoth_y=alpha*np.dot(x_t,sig(hdotx) -q)
	theta_new = theta - (axdoth_y/m)
	return theta_new

# read dataset
dataset = pd.read_csv('../data/1a.csv')

# defining x and quality data
q_data = dataset['quality']
x_data = dataset.iloc[:,0:11]

# defining some constants
col_list = list(x_data)
n_col = len(col_list)
m = len(dataset.index)
extra_col = np.ones((m,1))

# getting x and quality matrices
x_mat = np.asarray(x_data,dtype=float)
x_mat = np.append(extra_col,x_mat, axis=1)
q_mat = np.asarray(q_data, dtype=int)
theta = np.zeros(n_col+1, dtype = float)
theta[0]=1

# alpha value
alpha = 0.1

theta_new = update(x_mat,q_mat,theta,m, alpha)

print("regression running...")
# main loop for regression
while(cost(x_mat,q_mat,theta,m) - cost(x_mat,q_mat,theta_new,m)>0.0000001):
	theta=theta_new
	theta_new = update(x_mat, q_mat, theta_new,m,alpha)
print("theta values after custom logistic regression on whole data")
print(theta)



### 2.2 Scikit

print("scikit regression running...")
sk_logistic = LogisticRegression(penalty='none', solver ='saga')
model=sk_logistic.fit(x_mat,q_mat)
q_pred = model.predict(x_mat)
accuracy = metrics.accuracy_score(q_mat,q_pred)
print("Theta values from sciki-", model.coef_)


### 2.3 3-fold validation
print("3-fold validation running...")
trainx1= x_mat[0:1066]
testx1 = x_mat[1067:1599]
trainy1=q_mat[0:1066]
testy1 = q_mat[1067:1599]

# initialise theta
theta = np.zeros(n_col+1, dtype = float)
theta[0]=1
theta_new = update(trainx1,trainy1,theta,m, alpha)
# main loop for regression
while(cost(trainx1,trainy1,theta,m) - cost(trainx1,trainy1,theta_new,m)>0.0000001):
	theta=theta_new
	theta_new = update(trainx1, trainy1, theta_new,m,alpha)
# get the predicted arr
y_pred = sig(np.dot(testx1,np.transpose(theta)))
y_pred[y_pred>=0.5]=1
y_pred[y_pred<0.5]=0

# iteration 1
recall1=metrics.recall_score(testy1,y_pred)
precision1=metrics.precision_score(testy1,y_pred)
accuracy1 = metrics.accuracy_score(testy1,y_pred)

# iteration 2
trainx2= x_mat[534:1599]
testx2 = x_mat[0:533]
trainy2=q_mat[534:1599]
testy2 = q_mat[0:533]

theta = np.zeros(n_col+1, dtype = float)
theta[0]=1
theta_new = update(trainx2,trainy2,theta,m, alpha)
# main loop for regression
while(cost(trainx2,trainy2,theta,m) - cost(trainx2,trainy2,theta_new,m)>0.0000001):
	theta=theta_new
	theta_new = update(trainx2, trainy2, theta_new,m,alpha)

y_pred = sig(np.dot(testx2,np.transpose(theta)))

y_pred[y_pred>=0.5]=1
y_pred[y_pred<0.5]=0

recall2=metrics.recall_score(testy2,y_pred)
precision2=metrics.precision_score(testy2,y_pred)
accuracy2 = metrics.accuracy_score(testy2,y_pred)

# iteration 3

trainx3= np.concatenate((x_mat[0:533],x_mat[1066:1599]), axis = 0)
testx3 = x_mat[534:1066]
trainy3=np.concatenate((q_mat[0:533],q_mat[1066:1599]), axis = 0)
testy3 = q_mat[534:1066]

theta = np.zeros(n_col+1, dtype = float)
theta[0]=1
theta_new = update(trainx3,trainy3,theta,m, alpha)
while(cost(trainx3,trainy3,theta,m) - cost(trainx3,trainy3,theta_new,m)>0.0000001):
	theta=theta_new
	theta_new = update(trainx3, trainy3, theta_new,m,alpha)

y_pred = sig(np.dot(testx3,np.transpose(theta)))

y_pred[y_pred>=0.5]=1
y_pred[y_pred<0.5]=0

recall3=metrics.recall_score(testy3,y_pred)
precision3=metrics.precision_score(testy3,y_pred)
accuracy3 = metrics.accuracy_score(testy3,y_pred)

avg_recall = (recall1+recall2+recall3)/3
avg_precision = (precision1+precision2+precision3)/3
avg_accuracy = (accuracy1+accuracy2+accuracy3)/3
print("CUSTOM CLASSIFIER")
print("avg test accuracy for custom classifier=" + str(avg_accuracy*100))
print("avg test precision for custom classifier=" + str(avg_precision*100))
print("avg test recall for custom classifier=" + str(avg_recall*100))

# scikit for 2.3
# scikit iteration1
model1=sk_logistic.fit(trainx1,trainy1)
y_pred = model1.predict(testx1)
recall1=metrics.recall_score(testy1,y_pred)
precision1=metrics.precision_score(testy1,y_pred)
accuracy1 = metrics.accuracy_score(testy1,y_pred)
#scikit iteration 2
model2 = sk_logistic.fit(trainx2,trainy2)
y_pred = model2.predict(testx2)
recall2=metrics.recall_score(testy2,y_pred)
precision2=metrics.precision_score(testy2,y_pred)
accuracy2 = metrics.accuracy_score(testy2,y_pred)
# scikit iteration 3
model3 = sk_logistic.fit(trainx3,trainy3)
y_pred = model3.predict(testx3)
recall3=metrics.recall_score(testy3,y_pred)
precision3=metrics.precision_score(testy3,y_pred)
accuracy3 = metrics.accuracy_score(testy3,y_pred)

avg_recall = (recall1+recall2+recall3)/3
avg_precision = (precision1+precision2+precision3)/3
avg_accuracy = (accuracy1+accuracy2+accuracy3)/3
# print metrics
print("SCIKIT CLASSIFIER")
print("avg test accuracy for scikit classifier=" + str(avg_accuracy*100))
print("avg test precision for scikit classifier=" + str(avg_precision*100))
print("avg test recall for scikit classifier=" + str(avg_recall*100))
