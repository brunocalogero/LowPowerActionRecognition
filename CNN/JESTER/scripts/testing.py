import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt

#display options
pd.set_option('display.max_rows',None)
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(threshold=np.nan)


path1 = 'D:/JESTER/n_JESTER' 
path2 = 'D:/PIX2NVS/Events'


for class_index in ['Swiping_Left', 'Swiping_Right', 'Swiping_Up', 'Swiping_Down']:

	#create directory

	new_dirname = '{0}/n_Test/{1}'.format(path1, class_index)
	if not os.path.exists(new_dirname):
		os.makedirs(new_dirname)
		print("Directory ", new_dirname,  " created")
	else:
		print("Directory ", new_dirname,  " already exists")
	print("Looping over class: {0}".format(class_index))


	#for each class, convert raw NVS data to an interleaved 2 channel 176x100 matrix saved as .dat file

	for(dirpath, dirnames, filenames) in os.walk('{0}'.format(path2)):
		for counter, name in enumerate(filenames):
			if class_index in name:
				#print("Converting file {0}".format(counter))
				print('Loading file: {}'.format(name[:-8]))

				#load NVS text
				data = np.loadtxt(fname = '{0}/{1}'.format(path2,name))

				#create lists for x, y, t, p columns 
				data_x=data[:,0].astype(int)
				data_y=data[:,1].astype(int)
				data_t=data[:,2]
				data_p=data[:,3].astype(int)


				#create a time window t_r to t_ru between 40% and 60% of the total time length
				t_min = np.min(data_t)
				t_max = np.max(data_t)

				s=40 #window length parameter
				t_r = t_min + ((t_max - t_min) * (s / 100)) 
				t_ru = t_min + ((t_max - t_min) * ((100 - s) / 100)) 


				# index_need and index_need2 are the lower and upper indexes
				i_low = np.argmin(np.abs(data_t - t_r))
				i_up = np.argmin(np.abs(data_t - t_ru))

				x=[]
				y=[]
				p=[]

				for index in range(i_low,i_up):
					x.append(data_x[index])
					y.append(data_y[index])
					p.append(data_p[index])

				#transform categorical values to numerical
				for (i,item) in enumerate(p):
					if item == True:
						p[i] = 1
					elif item == False:
						p[i] = -1

				#create a dataframe out of the three lists
				df=pd.DataFrame({'x':x,'y':y,'p':p})

				#sort by x and y
				df1=df.sort_values(['x','y'])

				#divide data into two dataframes, for +ve and -ve values. 
				df1_neg = df1[df1.iloc[:,2]<0]
				df1_pos = df1[df1.iloc[:,2]>0]

				#group by x and y, sum polarities for each x-y coordinate, and display x,y,p and summed p
				df1_neg['sum_p'] = df1_neg.groupby(['x','y'])['p'].transform(sum)
				df1_pos['sum_p'] = df1_pos.groupby(['x','y'])['p'].transform(sum)

				#drop old p, drop duplicates, and reset dataframe index
				df_neg=df1_neg.drop(['p'],axis=1).drop_duplicates(subset=['x', 'y','sum_p']).reset_index(drop=True)
				df_pos=df1_pos.drop(['p'],axis=1).drop_duplicates(subset=['x', 'y','sum_p']).reset_index(drop=True)

				#convert negative values to absolute 
				df_neg['sum_p'] = df_neg['sum_p'].abs()

				#prepopulate 34x34x2 matrix with zeros (conversion to int32 for later use of fromfile)
				A = np.zeros(shape=(2,100,176), dtype=np.int32)

				#convert dataframe to np arrays B_neg and B_pos to allow easier indexing
				B_neg=df_neg.values
				B_pos=df_pos.values

				#B_pos -> A[1] and B_neg -> A[0]
				for row in B_neg:
					A[0][row[1]][row[0]]=row[2]
				for row in B_pos:
					A[1][row[1]][row[0]]=row[2]

				#interleaving A 2*100*176 -> c 100*352
				c=[]
				for row in range(A.shape[1]):
					result = [None]*(A.shape[2]*2)
					result[::2] = A[0][row]
					result[1::2] = A[1][row]
					c.append(result)
				numpy_c = np.array(c)

				#remove .mp4.txt
				filename_value = name[:-8]

				#save as .dat
				numpy_c.tofile('{0}/{1}.dat'.format(new_dirname, filename_value))