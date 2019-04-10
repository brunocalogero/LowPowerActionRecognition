import pandas as pd
from collections import namedtuple
import os
import numpy as np
from keras.preprocessing import image
from math import ceil
from multiprocessing import Process


datasetPath = 'C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/20bn-jester-v1'
train_file = 'C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/jester-v1-train.csv'
test_file = 'C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/jester-v1-validation.csv'
savePath = 'D:/JESTER/n_JESTER/Originaldataset'
resizePath = '{0}/Resized'.format(savePath)


def parse_10(train_file, test_file):

	d = pd.read_csv('{0}'.format(train_file)).values #resulting in 16725 videos for 4 labels
	#store indexes for the label
	tr1 = [item[0].split(';')[0] for item in d if "Swiping Left" in item[0]]
	tr2 = [item[0].split(';')[0] for item in d if "Swiping Right" in item[0]]
	tr3 = [item[0].split(';')[0] for item in d if "Swiping Up" in item[0]]
	tr4 = [item[0].split(';')[0] for item in d if "Swiping Down" in item[0]]
	tr5 = [item[0].split(';')[0] for item in d if "Sliding Two Fingers Left" in item[0]]
	tr6 = [item[0].split(';')[0] for item in d if "Sliding Two Fingers Right" in item[0]]
	tr7 = [item[0].split(';')[0] for item in d if "Sliding Two Fingers Up" in item[0]]
	tr8 = [item[0].split(';')[0] for item in d if "Sliding Two Fingers Down" in item[0]]
	tr9 = [item[0].split(';')[0] for item in d if "Zooming In With Two Fingers" in item[0]]
	tr10 = [item[0].split(';')[0] for item in d if "Zooming Out With Two Fingers" in item[0]]

	e = pd.read_csv('{0}'.format(test_file)).values  # resulting in 2008 videos for 4 labels
	# store indexes for the label
	te1 = [item[0].split(';')[0] for item in e if "Swiping Left" in item[0]]
	te2 = [item[0].split(';')[0] for item in e if "Swiping Right" in item[0]]
	te3= [item[0].split(';')[0] for item in e if "Swiping Up" in item[0]]
	te4 = [item[0].split(';')[0] for item in e if "Swiping Down" in item[0]]
	te5 = [item[0].split(';')[0] for item in e if "Sliding Two Fingers Left" in item[0]]
	te6 = [item[0].split(';')[0] for item in e if "Sliding Two Fingers Right" in item[0]]
	te7 = [item[0].split(';')[0] for item in e if "Sliding Two Fingers Up" in item[0]]
	te8 = [item[0].split(';')[0] for item in e if "Sliding Two Fingers Down" in item[0]]
	te9 = [item[0].split(';')[0] for item in e if "Zooming In With Two Fingers" in item[0]]
	te10 = [item[0].split(';')[0] for item in e if "Zooming Out With Two Fingers" in item[0]]

	Train = namedtuple('Train',['tr1','tr2','tr3','tr4','tr5','tr6','tr7','tr8','tr9','tr10'])
	Test = namedtuple('Test',['te1','te2','te3','te4','te5','te6','te7','te8','te9','te10'])
	tr,te = Train(tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10),Test(te1,te2,te3,te4,te5,te6,te7,te8,te9,te10)
	return tr,te

def create_dir(tr_or_te,i):
	# create Train or Test directory (tr_or_te = 'tr' or 'te')
	outputPath = 'D:/JESTER/n_JESTER/Originaldataset'
	class_index = classList[i]
	if(tr_or_te=='tr'):
		new_dirname = '{0}/Train/{1}'.format(outputPath, class_index)
		if not os.path.exists(new_dirname):
			os.makedirs(new_dirname)
			print("Directory ", new_dirname, " created")
		else:
			print("Directory ", new_dirname, " already exists")
	elif(tr_or_te=='te'):
		new_dirname = '{0}/Test/{1}'.format(outputPath, class_index)
		if not os.path.exists(new_dirname):
			os.makedirs(new_dirname)
			print("Directory ", new_dirname, " created")
		else:
			print("Directory ", new_dirname, " already exists")

def takespread(sequence, num):
	length = float(len(sequence))
	for i in range(num):
		yield sequence[int(ceil(i * length / num))]

def resize(path,resizedPath):
	for (dirpath, dirnames, filenames) in os.walk('{0}'.format(path)):
		for name in list(takespread(filenames, 12)):
			os.system("ffmpeg -i {0}/{1} -vf scale=176:100 {2}/{1}".format(path,name,resizedPath))

def save(resizedPath,savedPath):
	for (dirpath, dirnames, filenames) in os.walk('{0}'.format(resizedPath)):
		img_array=[]
		for counter, name in enumerate(filenames):
			img = image.img_to_array(image.load_img('{0}/{1}'.format(resizedPath,name),interpolation='bicubic'))
			img_array.append(img)
		#print(np.array(img_array).shape)
		np.save('{0}'.format(savedPath),img_array)


classList = ['Swiping_Left', 'Swiping_Right', 'Swiping_Up', 'Swiping_Down', 'Sliding_Two_Fingers_Left', 'Sliding_Two_Fingers_Right', 'Sliding_Two_Fingers_Up', 'Sliding_Two_Fingers_Down', 'Zooming_In_With_Two_Fingers', 'Zooming_Out_With_Two_Fingers']

def build(i):
	tr,te=parse_10(train_file, test_file)
	this_class=classList[i]

	for file in tr[i]:
		path = '{0}/{1}'.format(datasetPath, file)
		resizedPath = '{0}/Train/{1}/{2}'.format(resizePath, this_class, file)
		savedPath = '{0}/Train/{1}/{2}'.format(savePath,this_class,file)
		if not os.path.exists(resizedPath):
				os.makedirs(resizedPath)
		resize(path, resizedPath)
		save(resizedPath,savedPath)

	for file in te[i]:
		path = '{0}/{1}'.format(datasetPath, file)
		resizedPath = '{0}/Test/{1}/{2}'.format(resizePath, this_class, file)
		savedPath = '{0}/Test/{1}/{2}'.format(savePath,this_class,file)
		if not os.path.exists(resizedPath):
				os.makedirs(resizedPath)
		resize(path, resizedPath)
		save(resizedPath,savedPath)

if __name__ == '__main__':
	processes = []
	for i in range(10):
		create_dir('tr', i)
		create_dir('te', i)
		p = Process(target=build, args=(i,))
		processes.append(p)
		p.start()

	for process in processes:
		process.join()