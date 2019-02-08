import os
import cv2
from csv_parse import parse_swiping
from shutil import copyfile, copy2

"""
JESTER preprocessing using ffmpeg and PIX2NVS
"""

# JESTER CSV paths
train_file = 'C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/jester-v1-train.csv'
test_file = 'C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/jester-v1-validation.csv'
# Dataset paths
trainPath = 'D:/JESTER/n_JESTER/Train_files/16725/Videos'
storeTrainPath = 'D:/JESTER/n_JESTER/Train_files/16725/Events'
testPath = 'D:/JESTER/n_JESTER/Test_files/2008/Videos'
storeTestPath = 'D:/JESTER/n_JESTER/Test_files/2008/Events'
# PIX2NVS paths
inputPath = 'D:/PIX2NVS/input'
eventPath = 'D:/PIX2NVS/Events'

# convert folder of jpegs into mp4s for each of the 4 labels, for train files according to csv
def convert_train():
	swipe = parse_swiping(train_file, test_file)
	for item in swipe[0]:
		os.system("ffmpeg -f image2 -r 37 -i C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/20bn-jester-v1/{}/%5d.jpg D:/PIX2NVS/input/{};Swiping_Left.mp4".format(item,item))
	for item in swipe[1]:
		os.system("ffmpeg -f image2 -r 37 -i C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/20bn-jester-v1/{}/%5d.jpg D:/PIX2NVS/input/{};Swiping_Right.mp4".format(item,item))
	for item in swipe[2]:
		os.system("ffmpeg -f image2 -r 37 -i C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/20bn-jester-v1/{}/%5d.jpg D:/PIX2NVS/input/{};Swiping_Up.mp4".format(item,item))
	for item in swipe[3]:
		os.system("ffmpeg -f image2 -r 37 -i C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/20bn-jester-v1/{}/%5d.jpg D:/PIX2NVS/input/{};Swiping_Down.mp4".format(item,item))

# convert folder of jpegs into mp4s for each of the 4 labels, for test files according to csv
def convert_test():
	swipe = parse_swiping(train_file, test_file)
	for item in swipe[4]:
		os.system("ffmpeg -f image2 -r 37 -i C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/20bn-jester-v1/{}/%5d.jpg D:/JESTER/n_JESTER/Test_files/2008/Videos/{};Swiping_Left.mp4".format(item,item))
	for item in swipe[5]:
		os.system("ffmpeg -f image2 -r 37 -i C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/20bn-jester-v1/{}/%5d.jpg D:/JESTER/n_JESTER/Test_files/2008/Videos/{};Swiping_Right.mp4".format(item,item))
	for item in swipe[6]:
		os.system("ffmpeg -f image2 -r 37 -i C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/20bn-jester-v1/{}/%5d.jpg D:/JESTER/n_JESTER/Test_files/2008/Videos/{};Swiping_Up.mp4".format(item,item))
	for item in swipe[7]:
		os.system("ffmpeg -f image2 -r 37 -i C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/20bn-jester-v1/{}/%5d.jpg D:/JESTER/n_JESTER/Test_files/2008/Videos/{};Swiping_Down.mp4".format(item,item))

# rescale video size of all videos to a consistent 176:100
# parameter for setting train or test (tr_or_te = 'tr' or 'te')
def ffmpeg_rescale(tr_or_te,path,batch=None):
	if(tr_or_te=='tr'): # train files
		for (root, dirs, dat_files) in os.walk('{0}/{1}'.format(path, batch)):
			for file in dat_files:
				print(file)
				os.system('ffmpeg -i {2}/{1}/{0} -vf scale=176:100 {2}/resized_{1}/{0}'.format(file,batch,path))
	elif(tr_or_te=='te'): # test files
		for (root, dirs, dat_files) in os.walk(path):
			for file in dat_files:
				print(file)
				os.system('ffmpeg -i {1}/{0} -vf scale=176:100 {1}/resized_videos/{0}'.format(file,path))

# runs rescaling function for train and test
def run_rescale():
	batches = ['Batch_1', 'Batch_2', 'Batch_3', 'Batch_4'] # videos are stored in 4 batches because of PIX2NVS input limit of 5000 videos
	for batch in batches:
		print('Rescaling train batch: {}'.format(batch))
		ffmpeg_rescale('tr', trainPath, batch)
	print('Rescaling test batch')
	ffmpeg_rescale('te', testPath)

# deletes all files in folder - used when running PIX2NVS repeatedly
def delete_contents(folder):
	cnt = 0
	src_files = os.listdir(folder)
	for the_file in src_files:
		cnt += 1
		if (cnt % 400 == 0):
			print('Percent deleted: {}%'.format(round((cnt / len(src_files)) * 100, 2)))
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
		# elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
			print(e)

# copy and paste all files from source folder to destination folder
def copy(src,dest):
	src_files = os.listdir(src)
	cnt=0
	for file_name in src_files:
		cnt+=1
		if(cnt%400==0):
			print('Percent copied: {}%'.format(round((cnt/len(src_files))*100,2)))
		full_file_name = os.path.join(src, file_name)
		if (os.path.isfile(full_file_name)):
			copy2(full_file_name, dest)

# run PIX2NVS via cmd
def pix2nvs():
	os.system('D: & cd "/PIX2NVS" & PIX2NVS')

# run PIX2NVS function for train and test files, including copying and deleting back and forth
def convert_pix2nvs():

	print('Converting Train files')
	print('Deleting old PIX2NVS input')
	delete_contents(inputPath)
	print('Deleting PIX2NVS output')
	delete_contents(eventPath)
	batches = ['Batch_3', 'Batch_4']
	for batch in batches: #train
		print('Copying to PIX2NVS input')
		copy('{0}/resized_{1}'.format(trainPath,batch),inputPath)
		print('Running PIX2NVS')
		pix2nvs()
		print('Copying out of PIX2NVS events')
		copy(eventPath,storeTrainPath)
		print('Deleting PIX2NVS input')
		delete_contents(inputPath)
		print('Deleting PIX2NVS output')
		delete_contents(eventPath)
	print('Converting Test files') # test
	print('Copying to PIX2NVS input')
	copy(testPath, inputPath)
	print('Running PIX2NVS')
	pix2nvs()
	print('Copying out of PIX2NVS events')
	copy(eventPath, storeTestPath)
	print('Deleting PIX2NVS input')
	delete_contents(inputPath)
	print('Deleting PIX2NVS output')
	delete_contents(eventPath)


# RUN these:
#convert_train()
#convert_test()
#run_rescale()
#convert_pix2nvs()