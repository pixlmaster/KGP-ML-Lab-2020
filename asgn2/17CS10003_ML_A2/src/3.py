import pandas as pd
import numpy as np
import math
from sklearn import metrics 
from sklearn.tree import DecisionTreeClassifier
from numpy import log2 as log 
import pprint

# find the entropy of the current node
def find_entropy(df):
	# get input and output of current dataframe
	q_data = df.keys()[-1]
	x_data = df.keys()[:-1]
	# unique output list
	q_unique = df[q_data].unique()
	entropy=0
	m = len(df[q_data])
	no_q = df[q_data].value_counts()
	# sum entropy for each unique output
	for q in q_unique:
		numerator=no_q[q]
		f = numerator/m
		if f!=0:
			entropy += -f*np.log(f)
	return entropy

# find the entropy of the attributes
def find_entropy_attr(df,attr):
	q_data = df.keys()[-1]
	x_data = df.keys()[:-1]
	x_unique = df[attr].unique()
	q_unique = df[q_data].unique()
	entropy = 0
	# each unique element represents a new node
	for x in x_unique:
		t_entropy=0
		# total no of elements in this node
		denominator=len(df[attr][df[attr]==x])
		for q in q_unique:
			numerator = len(df[attr][df[attr]==x][df[q_data] == q])
			# numerator is number of elements with attribute matching x and output matching q
			f = numerator/(denominator)
			# use fraction to add entropy of each node when f!=0
			if(f!=0):
				t_entropy += -f*np.log(f)
		# weight it according to number of elements in node
		f2 = denominator/len(df)
		entropy += -f2*t_entropy
	# return the absolute value of the elements
	return abs(entropy)

	
# function to get the entropy
def max_entropy(df):
	q_data = df.keys()[-1]
	x_data = df.keys()[:-1]
	# make a list for gain
	gain =[]
	# get gain for each key
	for key in x_data:
		gain.append(find_entropy(df)-find_entropy_attr(df,key))
	# get the maximum gain
	max_index = np.argmax(gain)
	return x_data[max_index]

# get subtable with values equal to attr
def get_subtable(df, node,attr):
  return df[df[node] == attr].reset_index(drop=True)

# recursive function to build tree
def buildtree(df,tree=None):
	q_data = df.keys()[-1]
	x_data = df.keys()[:-1]

	features = len(x_data)
	# get the node with max entropy
	node = max_entropy(df)
	attr = np.unique(df[node])

	# initiliase tree on first iteration
	if tree is None:
		tree ={}
		tree[node] = {}

	for a in attr:
		# get the subtable with attribute a
		subtable = get_subtable(df,node,a)
		length = len(subtable)
		q_temp = subtable.keys()[-1]
		uniq = np.unique(subtable[q_temp])
		# termination condition
		if length<=10 or len(uniq)==1 or len(list(subtable))==2:
			# set the output as the element with max freq
			freq=subtable[q_temp].value_counts().idxmax()
			tree[node][a] = freq                                                 
		else:
			subtable = subtable.drop(node,axis=1)   
			# recursive call  
			tree[node][a] = buildtree(subtable) 
	# return the tree
	return tree

# classify the test data
def predict(row, tree):
	for column in list(row):
		if column in list(tree):
			try :
				result = tree[column][row[column]]
			except:
				return 1
			if isinstance(result,dict):
				return predict(row,result)
			else:
				return result

# read the 1b  dataframe
dataset = pd.read_csv('../data/1b.csv')
# some lists and dataframe definitions
q_data = dataset.keys()[-1]
x_list = dataset.keys()[:-1]
x_list = dataset[x_list]
q_list = dataset[q_data]
# convert datafram to dictionary
x_data = dataset.iloc[:,:-1].to_dict(orient = "records")

### 3.1 CUSTOM CLASSIFIER 

print("BUILDING CUSTOM CLASSIFIER...")
# build the tree
tree = buildtree(dataset )

print("BUILD COMPLETED")

pred = []
for row in x_data:
	pred.append(predict(row,tree))

# metrics calculated on whole of training data
print("train accuracy using custom classifier =" + str(100*metrics.accuracy_score(dataset[q_data],pred)))
print("train precision using custom classifier =" + str(100*metrics.precision_score(dataset[q_data],pred,average = 'macro')))
print("train recall using custom classifier =" + str(100*metrics.recall_score(dataset[q_data],pred,average = 'macro')))

### 3.2 USING SCIKIT

# using scikit classifier
print("SCIKIT CLASSIFIER RUNNING ...")
classify = DecisionTreeClassifier(criterion = 'entropy',min_samples_split=10)
sci_tree=classify.fit(x_list,dataset[q_data])
print("CLASSIFICATION COMPLETED")
sc_pred = sci_tree.predict(x_list)
# printing the metrics
print("train accuracy using scikit classifier =" + str(100*metrics.accuracy_score(dataset[q_data],sc_pred)))
print("train precision using scikit classifier =" + str(100*metrics.precision_score(dataset[q_data],sc_pred,average = 'macro')))
print("train recall using scikit classifier =" + str(100*metrics.recall_score(dataset[q_data],sc_pred,average = 'macro')))


### 3.3 3-FOLD CROSS VALIDATION
print("3F VALIDATION FOR CUSTOM CLASSIFIER...")
# 1st iteration
dataset1=dataset.iloc[0:1066]
tree1 = buildtree(dataset1)
testx1 = dataset.iloc[1066:1599].to_dict(orient = "records")
testy1 = q_list[1066:1599]

pred1 = []
for row in testx1:
	pred1.append(predict(row,tree1))

accuracy1=100*metrics.accuracy_score(testy1,pred1)
precision1=100*metrics.precision_score(testy1,pred1,average = 'macro')
recall1 = 100*metrics.recall_score(testy1,pred1,average = 'macro')

# 2nd iteration
dataset2=dataset.iloc[533:1599]
tree2 = buildtree(dataset2)
testx2 = dataset.iloc[0:533].to_dict(orient = "records")
testy2 = q_list[0:533]

pred2 = []
for row in testx2:
	pred2.append(predict(row,tree2))

accuracy2=100*metrics.accuracy_score(testy2,pred2)
precision2=100*metrics.precision_score(testy2,pred2,average = 'macro')
recall2 = 100*metrics.recall_score(testy2,pred2,average = 'macro')

# 3rd iteration
frames = [dataset.iloc[0:533],dataset.iloc[1066:1599]]
dataset3= pd.concat(frames)
tree3 = buildtree(dataset3)
testx3 = dataset.iloc[533:1066].to_dict(orient = "records")
testy3 = q_list[533:1066]

pred3 = []
for row in testx3:
	pred3.append(predict(row,tree3))

accuracy3=100*metrics.accuracy_score(testy3,pred3)
precision3=100*metrics.precision_score(testy3,pred3,average = 'macro')
recall3 = 100*metrics.recall_score(testy3,pred3,average = 'macro')

avg_accuracy = (accuracy1+accuracy2+accuracy3)/3
avg_precision = (precision1+precision2+precision3)/3
avg_recall = (recall1 +recall2 +recall3)/3

# print the metrics
print("avg test accuracy using custom classifier =" + str(avg_accuracy))
print("avg test precision using custom classifier =" + str(avg_precision))
print("avg test  recall using custom classifier =" + str(avg_recall))

print("3F VALIDATION FOR SCIKI CLASSIFIER...")

# 1st iteration
x_list1 = dataset1.keys()[:-1]
x_list1 = dataset1[x_list1]
testy1 = q_list[1066:1599]
testx1= x_list[1066:1599]

classify = DecisionTreeClassifier(criterion = 'entropy',min_samples_split=10)
sci_tree1=classify.fit(x_list1,dataset1[q_data])
sc_pred1 = sci_tree1.predict(testx1)

accuracy1=100*metrics.accuracy_score(testy1,sc_pred1)
precision1=100*metrics.precision_score(testy1,sc_pred1,average = 'macro')
recall1 = 100*metrics.recall_score(testy1,sc_pred1,average = 'macro')

# 2nd iteration
x_list2 = dataset2.keys()[:-1]
x_list2 = dataset2[x_list2]
testy2 = q_list[0:533]
testx2= x_list[0:533]


classify = DecisionTreeClassifier(criterion = 'entropy',min_samples_split=10)
sci_tree2=classify.fit(x_list2,dataset2[q_data])
sc_pred2 = sci_tree2.predict(testx2)

accuracy2=100*metrics.accuracy_score(testy2,sc_pred2)
precision2=100*metrics.precision_score(testy2,sc_pred2,average = 'macro')
recall2 = 100*metrics.recall_score(testy2,sc_pred2,average = 'macro')

# 3rd iteration
x_list3 = dataset3.keys()[:-1]
x_list3 = dataset3[x_list3]
testy3 = q_list[533:1066]
testx3= x_list[533:1066]

classify = DecisionTreeClassifier(criterion = 'entropy',min_samples_split=10)
sci_tree3=classify.fit(x_list3,dataset3[q_data])
sc_pred3 = sci_tree3.predict(testx3)

accuracy3=100*metrics.accuracy_score(testy3,sc_pred3)
precision3=100*metrics.precision_score(testy3,sc_pred3,average = 'macro')
recall3 = 100*metrics.recall_score(testy3,sc_pred3,average = 'macro')

avg_accuracy = (accuracy1+accuracy2+accuracy3)/3
avg_precision = (precision1+precision2+precision3)/3
avg_recall = (recall1 +recall2 +recall3)/3

# print the metrics
print("avg test accuracy using scikit classifier =" + str(avg_accuracy))
print("avg test precision using scikit classifier =" + str(avg_precision))
print("avg test  recall using scikit classifier =" + str(avg_recall))
