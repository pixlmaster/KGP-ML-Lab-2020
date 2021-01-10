import pandas as pd
import numpy as np
import math

def parse(x):
	first=x.split('_',1)[0]
	return first

#Opening the CSV file
csv_file = pd.read_csv('../data/data.csv')
csv_file = csv_file.drop(csv_file.index[13])
csv_file.iloc[:,0] = csv_file.iloc[:,0].apply(parse)

csv_file=csv_file.drop(csv_file.columns[0],axis='columns')
data = np.float64(np.array(csv_file.iloc[:,:].values))
rows =data.shape[0]
columns = data.shape[1]

print("running...")

for i in range(0,columns):
	freq = 0
	for j in range(0,rows):
		if(data[j][i] > 0):
			freq=freq+1
	for j in range(0,rows):
		tf = data[j][i]
		idf = math.log(1.0*(1 + rows)/(1+freq))
		data[j][i] = 1.0*tf*idf

for i in range(0,rows):
	magn = 0
	for j in range(0,columns):
		magn=magn+data[i][j]*data[i][j]
	magn=math.sqrt(magn)
	for j in range(0,columns):
		data[i][j]=data[i][j]/magn

pd.DataFrame(data).to_csv("../data/a.csv")
print("modified data saves as a.csv in data folder")