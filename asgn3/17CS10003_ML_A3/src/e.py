import pandas as pd
import numpy as np
import math

def parse(x):
	first=x.split('_',1)[0]
	return first

csv_file = pd.read_csv('../data/data.csv')
csv_file = csv_file.drop(csv_file.index[13])

rows =csv_file.shape[0]
columns = csv_file.shape[1]

csv_file.iloc[:,0] = csv_file.iloc[:,0].apply(parse)

c1 = csv_file.iloc[:,0]
c1 = np.asarray(c1)

c1[c1 == 'Buddhism'] = 0
c1[c1 == 'TaoTeChing'] = 1
c1[c1 == 'Upanishad'] = 2
c1[c1 == 'YogaSutra'] = 3
c1[c1 == 'BookOfProverb'] = 4
c1[c1 == 'BookOfEcclesiastes'] = 5
c1[c1 == 'BookOfEccleasiasticus'] = 6
c1[c1 == 'BookOfWisdom'] = 7

def c_entr(c):
	num=len(c)
	answer = (-num/rows)*math.log2(num/rows)
	return answer

def class_entropy(c):
	entropy=0
	for label in c:
		entropy+=c_entr(label)
	return entropy

def get_list(temptxt):
    final = []
    for i in range(0, 8):
        templist = list(map(int, temptxt.readline().split(',')))
        final.append(templist)
    return final

def cl_entr(c):
	num=len(c)
	answer = (-num/rows)*math.log2(num/rows)
	return answer

def cluster_entropy(c):
	entropy=0
	for label in c:
		entropy+=cl_entr(label)
	return entropy


def conditional_entropy(label,cluster_list,entr_class):
	answer=entr_class
	for cluster in cluster_list:
		total = len(cluster)
		entropy=0
		for i in range(0,8):
			temp=0
			for index in cluster:
				if(label[index]==i):
					temp=temp+1
			if(temp>0):
				entropy-= (temp/total)*math.log2(temp/total)
		answer -= (total/rows)*entropy
	return answer

def nmi(entr_class,file,label):
	cluster_list = get_list(file)
	entr_clust = cluster_entropy(cluster_list)
	entr_condn = conditional_entropy(label,cluster_list,entr_class)
	return (2*entr_condn)/(entr_class+entr_clust)


label_list =[]

for i in range(0,8):
	label_list.append(csv_file.iloc[:,0][csv_file.iloc[:,0] == i])

c_entropy= class_entropy(label_list)

print("NMI for agglomerative clustering")
agglotxt= open('../data/b.txt', 'r')
print(nmi(c_entropy,agglotxt,c1))

print("NMI for Kmeans clustering")
ktxt= open('../data/c.txt', 'r')
print(nmi(c_entropy,ktxt,c1))

print("NMI for agglomerative clustering after PCA")
agglotxt= open('../data/d1.txt', 'r')
print(nmi(c_entropy,agglotxt,c1))

print("NMI for Kmeans clustering after PCA")
ktxt= open('../data/d2.txt', 'r')
print(nmi(c_entropy,ktxt,c1))
