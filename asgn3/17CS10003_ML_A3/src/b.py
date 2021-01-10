import pandas as pd
import numpy as np
import math

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

def merge(c1,c2):
	n_vect = []
	n_vect.extend(c1.vectors)
	n_vect.append(c2.vectors)
	n_index =[]
	n_index.extend(c1.index)
	n_index.extend(c2.index)
	n_clust = cluster(n_vect,n_index)
	return n_clust


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

def init_dmat(cluster_len,dist_mat):
	for i in range(0,cluster_len):
		for j in range(0,cluster_len):
			if(i==j):
				dist_mat[i][j]=max_val
			else:
				dist_mat[i][j] = dist(csv_file[i],csv_file[j])

def main_loop(cluster_list,cluster_len,dist_mat):
	while(cluster_len>n_clusters):
		print("No of clusters=",cluster_len)
		min_d =max_val
		x=0
		y=0
		for i in range(0,cluster_len):
			for j in range(i+1,cluster_len):
				temp_dist=clust_dist(cluster_list[i],cluster_list[j],dist_mat)
				if(temp_dist<min_d):
					min_d = temp_dist
					x=cluster_list[i]
					y=cluster_list[j]
		cluster_list.remove(x)
		cluster_list.remove(y)
		new_clust = merge(x,y)
		cluster_list.append(new_clust)
		cluster_len-=1
	print("No of clusters=",cluster_len)

def data_write(cluster_list):
	file = open('../data/b.txt', 'w')
	for c in cluster_list:
	    for i in range(0, len(c.index)):
	        if(i == len(c.index ) -1 ):
	        	file.write(str(c.index[i]) + '\n')
	        else:
	        	file.write(str(c.index[i])+',')         
	file.close()

print("running...")

cluster_list = []

for i in range (0,rows):
	cluster_list.append(cluster([csv_file[i]],[i]))

cluster_len = len(cluster_list)

dist_mat = np.zeros((cluster_len,cluster_len))

init_dmat(cluster_len,dist_mat)

main_loop(cluster_list,cluster_len,dist_mat)

data_write(cluster_list)

print("file saved as b.txt in data folder")
