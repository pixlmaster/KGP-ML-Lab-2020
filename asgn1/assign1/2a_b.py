import matplotlib.pyplot as plt
import csv
import numpy as np

# cost function
def cost(x, y, theta ,x_mat, m, deg):
	res=np.dot(theta,x_mat) - y
	res_t =np.transpose(res)
	prod= np.dot(res,res_t)
	cost = np.asscalar(prod)/2
	cost=cost/m
	return cost
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

	# create theta and theta_new with all 0s as inititalisation

	print("estimated theta")
	print(theta)
	print("test error(on test set)=")
	print(cost(x_t,y_t,theta,x_mat_t,no_data_t,deg))

	train_error.append(cost(x,y,theta,x_mat,no_data,deg))
	test_error.append(cost(x_t,y_t,theta,x_mat_t,no_data_t,deg))

	# draw test x vs predicted y plots
	pred_y=np.dot(theta,x_mat_t)
	plt.scatter(x_t, pred_y, label= "points", color= "green",marker= ".", s=1) 
	plt.xlabel('test x - axis') 
	plt.ylabel(' predicted y - axis')
	tit= "predicted values for degree=" + str(deg) 
	plt.title(tit) 
	plt.legend() 
	plt.show()

 #plot train error
plt.plot(deg_arr, train_error, label = "train error")
#plot test error
plt.plot(deg_arr, test_error, label = "test error")
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis')
# giving title
plt.title('Train error / test error vs degree of polynomial')
# show a legend on the plot 
plt.legend() 

# find the minimum training error polynomial's degreee
min_train=test_error[0]
min_degree=1
count=1
for itr in test_error:
	if itr<min_train:
		min_train=itr
		min_degree=count
	count=count+1

print("most suitable value for n is" + str(min_degree) + ", since the test error is smallest here")

# function to show the plot 
plt.show()