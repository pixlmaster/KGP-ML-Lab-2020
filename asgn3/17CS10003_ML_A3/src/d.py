import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import math
import random

csv_file = pd.read_csv('../data/a.csv')
csv_file=csv_file.drop(csv_file.columns[0],axis='columns')

csv_file = csv_file.to_numpy()

pca = PCA(n_components = 100)
csv_file = pca.fit_transform(csv_file)

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

def main_loop_kmeans(centroids, asgn_class):
	for i in range(0,1000):
		print("iteration no =", i)
		for j in range(0, rows):
			asgn_class[j] = center_dist(centroids,csv_file[j])
		update_centroid(centroids,asgn_class)

def main_loop_agglo(cluster_list,cluster_len,dist_mat):
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


def data_write_kmeans(asgn_class):
	file= open('../data/d2.txt', 'w')
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

def data_write_agglo(cluster_list):
	file = open('../data/d1.txt', 'w')
	for c in cluster_list:
	    for i in range(0, len(c.index)):
	        if(i == len(c.index ) -1 ):
	        	file.write(str(c.index[i]) + '\n')
	        else:
	        	file.write(str(c.index[i])+',')         
	file.close()

asgn_class = np.zeros((rows,1))

centroids = np.zeros((8,columns))
random_centroid(centroids)
main_loop_kmeans(centroids,asgn_class)
data_write_kmeans(asgn_class)

cluster_list = []

for i in range (0,rows):
	cluster_list.append(cluster([csv_file[i]],[i]))

cluster_len = len(cluster_list)

dist_mat = np.zeros((cluster_len,cluster_len))

init_dmat(cluster_len,dist_mat)

main_loop_agglo(cluster_list,cluster_len,dist_mat)

data_write_agglo(cluster_list)

print("agglomerative save in d1.txt and kmeans in d2.txt")
