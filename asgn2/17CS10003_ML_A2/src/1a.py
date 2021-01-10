import pandas as pd

#Opening the CSV file
csv_file = pd.read_csv('../data/winequality-red.csv', sep=';')

# alter the quality column
csv_file.loc[csv_file['quality']<=6,'quality']=0
csv_file.loc[csv_file['quality']>6,'quality']=1

# fidn the max and min fo each column
max_col=csv_file.max(axis = 0)
min_col=csv_file.min(axis = 0)

# get the headers of the csv_file
col_list=list(csv_file)

# alter the remaining columns
for i in range(len(col_list)):
	# Alter all columns except quality
	if(col_list[i]!='quality'):
		csv_file[col_list[i]]=(csv_file[col_list[i]] - min_col[i])/(max_col[i] - min_col[i])

# save the file
csv_file.to_csv('../data/1a.csv', index=False)
print("New file written successfully")