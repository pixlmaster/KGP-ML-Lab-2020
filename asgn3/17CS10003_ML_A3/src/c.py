import pandas as pd
import numpy as np
import math
import random

csv_file = pd.read_csv('../data/a.csv')
csv_file=csv_file.drop(csv_file.columns[0],axis='columns')

csv_file = csv_file.to_numpy()

rows =csv_file.shape[0]
columns = csv_file.shape[1]

max_val = 1e10
n_clusters = 8

class cluster:
	def __init__(self,vectors,index):
		self.vectors = vectors
		self.index = index

def clust_dist( c1, c2, dist_mat):
	n_c1 = len(c1.index)
	n_c2 = len(c2.index)
	m_dist = max_val
	for i in range(0,n_c1):
		for j in range(0,n_c2):
			temp_dist =dist_mat[c1.index[i]][c2.index[j]]
			if (temp_dist<m_dist):
				m_dist=temp_dist
	return m_dist

def dist(x,y):
	dot = np.dot(x,y.transpose())
	magnx = np.dot(x,x.transpose())
	magnx = math.sqrt(magnx)
	magny = np.dot(y,y.transpose())
	magny = math.sqrt(magny)
	distance = dot/(magnx*magny)
	return math.exp(-1.0*distance)

def center_dist(centroids, c):
	mind=max_val
	index=0
	for i in range(0,8):
		temp = dist(centroids[i],c)
		if temp<mind:
			mind=temp
			index = i
	return index

def init_dmat(cluster_len,dist_mat):
	for i in range(0,cluster_len):
		for j in range(0,cluster_len):
			if(i==j):
				dist_mat[i][j]=max_val
			else:
				dist_mat[i][j] = dist(csv_file[i],csv_file[j])

def random_centroid(centroids):
	indexes = random.sample(range(0,rows),n_clusters)
	for i in range(n_clusters):
		centroids[i]=csv_file[indexes[i]]

def update_centroid(centroids,asgn_class):
	for c in range(0,8):
		temp_row = np.zeros((1, columns))
		freq=0
		for i in range(0, rows):
			if(asgn_class[i] == c):
				temp_row+=csv_file[i]
				freq+=1
		centroids[c] = temp_row/freq

def main_loop(centroids, asgn_class):
	for i in range(0,1000):
		print("iteration no =", i)
		for j in range(0, rows):
			asgn_class[j] = center_dist(centroids,csv_file[j])
		update_centroid(centroids,asgn_class)


def data_write(asgn_class):
	file= open('../data/c.txt', 'w')
	for c in range(0, 8):
	    count = 0
	    for i in range(0, rows):
	        if(asgn_class[i] == c):
	            if(count == 0):
	                file.write(str(i));
	                count = 1
	            else:
	            	file.write(',' + str(i))
	    file.write('\n')
	file.close()

print("running...")

asgn_class = np.zeros((rows,1))

centroids = np.zeros((8,columns))
random_centroid(centroids)
main_loop(centroids,asgn_class)
data_write(asgn_class)

print("data saved in c.txt")
