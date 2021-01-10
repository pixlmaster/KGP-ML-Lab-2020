import matplotlib.pyplot as plt
import csv
import numpy as np

# cost function
def cost(x, y, theta ,x_mat, m, deg):
	res=np.dot(theta,x_mat) - y
	res_t = np.transpose(res)
	prod= np.dot(res,res_t)
	cost = np.asscalar(prod)/2
	cost=cost/m
	return cost

def ridge_cost(x, y, theta ,x_mat, m, deg, lamd):
	res=np.dot(theta,x_mat) - y
	res_t = np.transpose(res)
	prod= np.dot(res,res_t)
	# extra step for ridge
	prod = prod + lamd*np.sum(theta**2)
	cost = np.asscalar(prod)/2
	cost=cost/m
	return cost

def lasso_cost(x, y, theta ,x_mat, m, deg, lamd):
	res=np.dot(theta,x_mat) - y
	res_t = np.transpose(res)
	prod= np.dot(res,res_t)
	# extra step for lasso
	prod = prod + lamd*np.sum(theta)
	cost = np.asscalar(prod)/2
	cost=cost/m
	return cost

def lasso_update(x, y, theta, x_mat, m, deg, alpha, lamd):
	inner_prod = np.dot(theta,x_mat) - y
	x_mat_trans = np.transpose(x_mat)
	outer_prod = np.dot(inner_prod,x_mat_trans)
	# extra step for lasso
	outer_prod = outer_prod + lamd
	outer_prod=outer_prod*alpha
	outer_prod=outer_prod/m
	return(theta - outer_prod)
	
def ridge_update(x, y, theta, x_mat, m, deg, alpha, lamd):
	inner_prod = np.dot(theta,x_mat) - y
	x_mat_trans = np.transpose(x_mat)
	outer_prod = np.dot(inner_prod,x_mat_trans)
	# extra step for ridge
	outer_prod = outer_prod + lamd*theta
	outer_prod=outer_prod*alpha
	outer_prod=outer_prod/m
	return(theta - outer_prod)

#update the value of theta
def update(x, y, theta, x_mat, m, deg, alpha):
	inner_prod = np.dot(theta,x_mat) - y
	x_mat_trans = np.transpose(x_mat)
	outer_prod = np.dot(inner_prod,x_mat_trans)
	outer_prod=outer_prod*alpha
	outer_prod=outer_prod/m
	return(theta - outer_prod)
# defining empty arrays
temp_x=[]
temp_y=[]
temp_z=[]
temp_x_t=[]
temp_y_t=[]
temp_z_t=[]

# openeing train.csv
with open('train.csv') as csv_file:
	csv_reader = csv.reader(csv_file , delimiter=',')
	line_no=0
	for row in csv_reader:
		if(line_no!=0):
			temp_x.append(float(row[0]))
			temp_y.append(float(row[1]))
			temp_z.append(1)
		line_no=line_no+1
# creating numpy arrays of train.csv
x=np.asarray(temp_x, dtype = float)
y=np.asarray(temp_y, dtype = float)
x_mat=np.asarray(temp_z, dtype=float)

#opening test.csv
with open('test.csv') as csv_file_t:
	csv_reader_t = csv.reader(csv_file_t , delimiter=',')
	line_no=0
	for row in csv_reader_t:
		if(line_no!=0):
			temp_x_t.append(float(row[0]))
			temp_y_t.append(float(row[1]))
			temp_z_t.append(1)
		line_no=line_no+1
# creating numpy arrays of test.csv
x_t=np.asarray(temp_x_t, dtype = float)
y_t=np.asarray(temp_y_t, dtype = float)
x_mat_t=np.asarray(temp_z_t, dtype=float)

#some constants
no_data = len(x)
no_data_t= len(x_t)
alpha=0.05

train_error=[]
test_error=[]
deg_arr=[]

# loop for every degree polynomial from 1-9
for itr in range(9):
	# read theta values from theta.csv
	deg=itr+1
	file1=open('theta.csv')
	freader = csv.reader(file1 , delimiter=',')
	line_no=0
	theta=[]
	for row in freader:
		#print(line_no)
		if line_no==itr:
			for itr1 in row:
				theta.append(float(itr1))
		line_no=line_no+1

	file1.close()
	#append current degree to deg arr
	deg_arr.append(deg)
	# append row to x matrix
	temp=[]
	for itr1 in range(no_data):
		temp.append(pow(x[itr1],deg))
	new_row=np.asarray(temp, dtype= float)
	x_mat= np.vstack([x_mat,new_row])

	# append row to test x matrix
	temp=[]
	for itr1 in range(no_data_t):
		temp.append(pow(x_t[itr1],deg))	
	new_row_t=np.asarray(temp, dtype= float)
	x_mat_t=np.vstack([x_mat_t,new_row_t])
	# populate training set error and test error
	print("estimated theta")
	print(theta)
	print("test error(on test set)=")
	print(cost(x_t,y_t,theta,x_mat_t,no_data_t,deg))
	train_error.append(cost(x,y,theta,x_mat,no_data,deg))
	test_error.append(cost(x_t,y_t,theta,x_mat_t,no_data_t,deg))
# find the minimum training error polynomial's degreee
min_train=train_error[0]
min_degree=1
count=1
for itr in train_error:
	if itr<min_train:
		min_train=itr
		min_degree=count
	count=count+1
# find the maximum training error polynomial's degreee
max_train=train_error[0]
max_degree=1
count=1
for itr in train_error:
	if itr>max_train:
		max_train=itr
		max_degree=count
	count=count+1
#lasso regression for min
x_mat=np.asarray(temp_z, dtype=float)
x_mat_t=np.asarray(temp_z_t, dtype=float)

# initilasing x matrices for training set
for itr1 in range(min_degree):
	temp=[]
	for itr2 in range(no_data):
		temp.append(pow(x[itr2],itr1+1))
	new_row=np.asarray(temp, dtype= float)
	x_mat= np.vstack([x_mat,new_row])

# initilasing x matrices for test set
for itr1 in range(min_degree):
	temp=[]
	for itr2 in range(no_data_t):
		temp.append(pow(x_t[itr2],itr1+1))
	new_row=np.asarray(temp, dtype= float)
	x_mat_t= np.vstack([x_mat_t,new_row])

lam_arr=[0.25,0.5,0.75,1]
ltrain_cost=[]
ltest_cost=[]
# lasso regression on min poly begin
for itr in range(1,5):
	lamd=itr/4
	# initial conditions
	temp=[]
	for itr1 in range(min_degree+1):
		temp.append(0)
	theta = np.asarray(temp, dtype = float)
	theta_new = lasso_update(x, y, theta, x_mat, no_data, min_degree, alpha,lamd)

	# loop for regression
	iters=0
	while(iters<500000):
		theta= theta_new
		theta_new = lasso_update(x, y, theta_new, x_mat, no_data, min_degree, alpha,lamd)
		iters=iters+1
	ltrain_cost.append(lasso_cost(x,y,theta,x_mat,no_data,min_degree,lamd))
	ltest_cost.append(lasso_cost(x_t,y_t,theta,x_mat_t,no_data_t,min_degree,lamd))
# printing the req params
print("estimated theta(min train error poly)")
print(theta)
print("test error(on test set)=")
print(lasso_cost(x_t,y_t,theta,x_mat_t,no_data_t,min_degree,lamd))

#plot train error
plt.plot(lam_arr, ltrain_cost, label = "laso train error")
#plot test error
plt.plot(lam_arr, ltest_cost, label = "lasso test error")
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis')
# giving title
plt.title('lasso Train error / test error vs lambda values(min train error poly)')
# show a legend on the plot 
plt.legend() 

# function to show the plot 
plt.show()

#lasso regression for max
x_mat=np.asarray(temp_z, dtype=float)
x_mat_t=np.asarray(temp_z_t, dtype=float)

# initialising x matrix for training set
for itr1 in range(max_degree):
	temp=[]
	for itr2 in range(no_data):
		temp.append(pow(x[itr2],itr1+1))
	new_row=np.asarray(temp, dtype= float)
	x_mat= np.vstack([x_mat,new_row])


for itr1 in range(max_degree):
	temp=[]
	for itr2 in range(no_data_t):
		temp.append(pow(x_t[itr2],itr1+1))
	new_row=np.asarray(temp, dtype= float)
	x_mat_t= np.vstack([x_mat_t,new_row])

# lasso begin
lam_arr=[0.25,0.5,0.75,1]
ltrain_cost=[]
ltest_cost=[]
for itr in range(1,5):
	lamd=itr/4
	
	temp=[]
	for itr1 in range(max_degree+1):
		temp.append(0)
	theta = np.asarray(temp, dtype = float)
	theta_new = lasso_update(x, y, theta, x_mat, no_data, max_degree, alpha,lamd)

	# loop for regression
	iters=0
	while(iters<500000):
		theta= theta_new
		theta_new = lasso_update(x, y, theta_new, x_mat, no_data, max_degree, alpha,lamd)
		iters=iters+1
	ltrain_cost.append(lasso_cost(x,y,theta,x_mat,no_data,max_degree,lamd))
	ltest_cost.append(lasso_cost(x_t,y_t,theta,x_mat_t,no_data_t,max_degree,lamd))
# print important params
print("estimated theta(max train error poly)")
print(theta)
print("test error(on test set)=")
print(lasso_cost(x_t,y_t,theta,x_mat_t,no_data_t,max_degree,lamd))

#plot train error
plt.plot(lam_arr, ltrain_cost, label = "laso train error")
#plot test error
plt.plot(lam_arr, ltest_cost, label = "lasso test error")
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis')
# giving title
plt.title('lasso Train error / test error vs lambda values(max train error poly)')
# show a legend on the plot 
plt.legend() 

# function to show the plot 
plt.show()

#ridge regression for min
x_mat=np.asarray(temp_z, dtype=float)
x_mat_t=np.asarray(temp_z_t, dtype=float)

# initilasing x matrices for training set
for itr1 in range(min_degree):
	temp=[]
	for itr2 in range(no_data):
		temp.append(pow(x[itr2],itr1+1))
	new_row=np.asarray(temp, dtype= float)
	x_mat= np.vstack([x_mat,new_row])

# initilasing x matrices for test set
for itr1 in range(min_degree):
	temp=[]
	for itr2 in range(no_data_t):
		temp.append(pow(x_t[itr2],itr1+1))
	new_row=np.asarray(temp, dtype= float)
	x_mat_t= np.vstack([x_mat_t,new_row])

# ridge regression on min poly begin
lam_arr=[0.25,0.5,0.75,1]
ltrain_cost=[]
ltest_cost=[]
for itr in range(1,5):
	lamd=itr/4
	
	temp=[]
	for itr1 in range(min_degree+1):
		temp.append(0)
	theta = np.asarray(temp, dtype = float)
	theta_new = ridge_update(x, y, theta, x_mat, no_data, min_degree, alpha,lamd)

	# loop for regression
	iters=0
	while(iters<500000):
		theta= theta_new
		theta_new = ridge_update(x, y, theta_new, x_mat, no_data, min_degree, alpha,lamd)
		iters=iters+1
	ltrain_cost.append(ridge_cost(x,y,theta,x_mat,no_data,min_degree,lamd))
	ltest_cost.append(ridge_cost(x_t,y_t,theta,x_mat_t,no_data_t,min_degree,lamd))
# printing the req params
print("estimated theta(min train error poly)")
print(theta)
print("test error(on test set)=")
print(ridge_cost(x_t,y_t,theta,x_mat_t,no_data_t,min_degree,lamd))

#plot train error
plt.plot(lam_arr, ltrain_cost, label = "ridge train error")
#plot test error
plt.plot(lam_arr, ltest_cost, label = "ridge test error")
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis')
# giving title
plt.title('ridge Train error / test error vs lambda values(min train error poly)')
# show a legend on the plot 
plt.legend() 

# function to show the plot 
plt.show()

#ridge regression for max
x_mat=np.asarray(temp_z, dtype=float)
x_mat_t=np.asarray(temp_z_t, dtype=float)

# initialising x matrix for training set
for itr1 in range(max_degree):
	temp=[]
	for itr2 in range(no_data):
		temp.append(pow(x[itr2],itr1+1))
	new_row=np.asarray(temp, dtype= float)
	x_mat= np.vstack([x_mat,new_row])

# initialising x matrix for test set
for itr1 in range(max_degree):
	temp=[]
	for itr2 in range(no_data_t):
		temp.append(pow(x_t[itr2],itr1+1))
	new_row=np.asarray(temp, dtype= float)
	x_mat_t= np.vstack([x_mat_t,new_row])

# ridge begin
lam_arr=[0.25,0.5,0.75,1]
ltrain_cost=[]
ltest_cost=[]
for itr in range(1,5):
	lamd=itr/4
	
	temp=[]
	for itr1 in range(max_degree+1):
		temp.append(0)
	theta = np.asarray(temp, dtype = float)
	theta_new = ridge_update(x, y, theta, x_mat, no_data, max_degree, alpha,lamd)

	# loop for regression
	while(iters<500000):
		theta= theta_new
		theta_new = ridge_update(x, y, theta_new, x_mat, no_data, max_degree, alpha,lamd)
		iters=iters+1
	ltrain_cost.append(ridge_cost(x,y,theta,x_mat,no_data,max_degree,lamd))
	ltest_cost.append(ridge_cost(x_t,y_t,theta,x_mat_t,no_data_t,max_degree,lamd))
# print important params
print("estimated theta(max error polynomial)")
print(theta)
print("test error(on test set)=")
print(ridge_cost(x_t,y_t,theta,x_mat_t,no_data_t,min_degree,lamd))

#plot train error
plt.plot(lam_arr, ltrain_cost, label = "ridge train error")
#plot test error
plt.plot(lam_arr, ltest_cost, label = "ridge test error")
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis')
# giving title
plt.title('ridge Train error / test error vs lambda values(max train error poly)')
# show a legend on the plot 
plt.legend() 

# function to show the plot 
plt.show()

print("Both of the regularisation are not good but if we were to choose one We will prefer Lasso over Ridge regularisation because the error in lasso is considerably less than over in ridge")