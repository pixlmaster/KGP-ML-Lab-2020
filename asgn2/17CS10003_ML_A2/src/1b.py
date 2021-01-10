import pandas as pd
import numpy as np

#Opening the CSV file
csv_file = pd.read_csv('../data/winequality-red.csv', sep=';')

# alter the quality column
csv_file.loc[csv_file['quality']<5,'quality']=0
csv_file.loc[(csv_file['quality']==5) | (csv_file['quality']==6),'quality']=1
csv_file.loc[csv_file['quality']>6, 'quality'] =2

# get the headers of the csv_file
col_list = list(csv_file)

# fidn the mean and standard deviation of each column
mean_list = csv_file.mean(axis=0)
dev_list = csv_file.std(axis=0)

# Normalize
for i in range(len(col_list)):
	# Alter all columns except quality
	if(col_list[i]!='quality'):
		csv_file[col_list[i]]=(csv_file[col_list[i]] - mean_list[i])/dev_list[i]

min_list = csv_file.min(axis=0)
max_list = csv_file.max(axis=0)
size_list = (max_list-min_list)/4

# alter the remaining columns into buckets
# Note the max element has been kept in bucket 3 as it is a corner case i.e the last set inclusive at both ends
for i in range(len(col_list)):
	# Alter all columns except quality
	if(col_list[i]!='quality'):
		csv_file[col_list[i]]=np.floor((csv_file[col_list[i]] - min_list[i])/size_list[i])
		csv_file[col_list[i]] = csv_file[col_list[i]].astype(int)
		csv_file.loc[csv_file[col_list[i]]==4,col_list[i]]=3

#print(csv_file.head(20))

# save the file
csv_file.to_csv('../data/1b.csv', index=False)
print("New file written successfully")
