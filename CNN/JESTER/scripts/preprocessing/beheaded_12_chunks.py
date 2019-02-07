import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import cv2
import dlib
import time
np.set_printoptions(threshold=10)

# Set paths
outputPath = 'D:/JESTER/n_JESTER'
videoPath_train = 'D:/JESTER/n_JESTER/Train_files/16725/Videos'
videoPath_test = 'D:/JESTER/n_JESTER/Test_files/2008/Videos'
eventsPath_train = 'D:/JESTER/n_JESTER/Train_files/16725/Events'
eventsPath_test = 'D:/JESTER/n_JESTER/Test_files/2008/Events'
classList = ['Swiping_Left', 'Swiping_Right', 'Swiping_Up', 'Swiping_Down']

# Set detector (dlib HOG)
detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

def create_dir(tr_or_te):
	# create Train or Test directory (tr_or_te = 'tr' or 'te')
	for class_index in classList:
		if(tr_or_te=='tr'):
			new_dirname = '{0}/n_Train_beheaded/{1}'.format(outputPath, class_index)
			if not os.path.exists(new_dirname):
				os.makedirs(new_dirname)
				print("Directory ", new_dirname, " created")
			else:
				print("Directory ", new_dirname, " already exists")
		elif(tr_or_te=='te'):
			new_dirname = '{0}/n_Test_beheaded/{1}'.format(outputPath, class_index)
			if not os.path.exists(new_dirname):
				os.makedirs(new_dirname)
				print("Directory ", new_dirname, " created")
			else:
				print("Directory ", new_dirname, " already exists")

def load_vid(videoPath):
    # Outputs frames from mp4
	cap = cv2.VideoCapture(videoPath)

	frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
	fc, ret = 0, True

	while (fc < frameCount and ret):
		ret, buf[fc] = cap.read()
		fc += 1
	cap.release()
	return buf

def visualizor(img, box):
    # Visualizes detection box on frame
	win.clear_overlay()
	win.set_image(img)
	win.add_overlay(box)
	time.sleep(0.125)

def chunkify(x):
	# x denotes number of chunks created between 0% and 100% of the NVS video
	# e.g. the output if x=3 is (0, 33.33), (33.33, 66.67), (66.667, 100.0)
	ts = [[]]
	ts.append([])
	t0, t1 = 0, 100 / x
	ts[0].append(round(t0, 3)), ts[1].append(round(t1, 3))
	for i in range(x - 1):
		t0 += (100 / x)
		t1 += (100 / x)
		ts[0].append(round(t0, 3)), ts[1].append(round(t1, 3))
	return ts
def load_events(eventsPath,number_of_frames):
	# Creates an array from events text whose shape equals the frame count of original video (usually 37)

	data = np.loadtxt(eventsPath)
	# create lists for x, y, t, p columns
	data_x = data[:, 0].astype(int)
	data_y = data[:, 1].astype(int)
	data_t = data[:, 2]
	data_p = data[:, 3].astype(int)

	# create a time window t_r to t_ru between 40% and 60% of the total time length
	t_min = np.min(data_t)
	t_max = np.max(data_t)

	ts = chunkify(x=number_of_frames) # default val 37 corresponds to # of frames in original video
	event_chunks = []

	for cnt, (r, s) in enumerate(zip(ts[0], ts[1])):
		t_r = t_min + ((t_max - t_min) * (r / 100))
		t_ru = t_min + ((t_max - t_min) * ((s) / 100))

		# index_need and index_need2 are the lower and upper indexes
		i_low = np.argmin(np.abs(data_t - t_r))
		i_up = np.argmin(np.abs(data_t - t_ru))

		# create x, y, p lists using indexes that correspond with the particular chunk
		x, y, p, t = [], [], [], []
		for index in range(i_low, i_up):
			x.append(data_x[index])
			y.append(data_y[index])
			p.append(data_p[index])
			t.append(data_t[index])

		# convert polarities that are zeroes to negative so that they can be stacked
		arr = np.array(p)
		arr[arr == 0] = -1
		p = list(arr)

		# create a dataframe out of the three lists and sort by x and y
		df = pd.DataFrame({'x': x, 'y': y, 't': t, 'p': p}).sort_values(['x', 'y'])

		event_chunks.append(df.values)
	event_chunks = np.array(event_chunks)

	return event_chunks

def detect(videoPath, box_expander=10, visualize=False):
	# Detects face in mp4 video

	buf = load_vid(videoPath)
	all_box, all_conf = [], []

	for j in range(buf.shape[0]):
		img = buf[j]
		box, conf, idx = detector.run(img, 1, -1)
		if (visualize == True):
			visualizor(img, box)

		if not box: # empty detection
			all_box.append(box) # empty box is a rectangles[]
			all_conf.append(None) # empty conf is a []. Appending an empty list causes numpy array issues so we append None
		elif box: # non-empty detection
			# remove potential second detection (as they are usually erroneous) by indexing for first detection
			box, conf = np.array(box)[0], np.array(conf)[0]
			# convert from dlib rect to np for easier handling
			box = [(box.left()) - box_expander, (box.right()) + box_expander, (box.top()) - box_expander, (box.bottom()) + box_expander]  # xmin, xmax, ymin, ymax (top and bottom are reversed for some reason)
			all_box.append(box)
			all_conf.append(conf)

	if not all_box:
		print('No face found')
		return  # exits this function if no face is detected
	else:
		return np.array(all_box), np.array(all_conf)

def behead(videoPath, eventsPath, conf_threshold=-0.5, visualize=False):
	# removes events within facial region if detection is True
	# box: Rectangle boundaries that contain face. Shape: (37, 4) (xmin, xmax, ymin, ymax)
	# conf: The score is bigger for more confident detections. Shape: (37,)
	# events: NVS events divided into 37 chunks (37 is the number of frames in the original video)
	# conf_threshold: Hardcoded to -0.5. Higher it is, the more selective the face remover is.

	detection = detect(videoPath, visualize)
	if detection is None:
		return
	else:
		box, conf = detection[0], detection[1]
		events = load_events(eventsPath,number_of_frames=len(conf))
		z, z2 = 0, 0  # miscellaneous counters
		removal = []
		beheaded_events = []
		for cnt, chunk in enumerate(events):
			temp = []
			z1 = 0  # counter
			if conf[cnt] is None:
				continue
			else:
				xmin, xmax, ymin, ymax = box[cnt][0], box[cnt][1], box[cnt][2], box[cnt][3]
				for row, (x, y) in enumerate(chunk[:, 2:]):  # for each chunk corresponding with that frame (x=37)
					# for removal of face, x and y must lie within the rectangle, and detection confidence should exceed threshold (i.e. hand is not covering face)
					if (conf[cnt] > conf_threshold and xmin <= x <= xmax and ymin <= y <= ymax):
						temp.append(row) # collect indices of rows to be deleted
						# print(events[cnt][row]) # works
						z += 1
						z1 += 1
				z2 += len(chunk)
				removal.append(temp) # collect indices of rows to be deleted, per chunk
				#print('# of face coords in frame {2} = {0} out of {1}'.format(z1, len(chunk), cnt))

		print('Total # of events removed = {0} out of {1}'.format(z, z2))
		for cnt in range(len(removal)):
			index = removal[cnt]
			beheaded_chunk = np.delete(events[cnt], index, axis=0)  # delete specified row from events
			beheaded_events.append(beheaded_chunk)
		beheaded_events = [beheaded_events[i].tolist() for i in range(len(beheaded_events))]  # converts sublist to list
		beheaded_events = np.array([item for sublist in beheaded_events for item in sublist])
		return beheaded_events

def beheaded_to_A(videoPath,eventsPath,number_of_chunks):
	# For beheaded videos, convert events to matrix

	beheaded_events = behead(videoPath,eventsPath)
	if (beheaded_events.shape[0] == 0): # check if list is empty (if not beheaded_events caused a problem with else condition)
		return
	else:
		data=beheaded_events #unpacks from (37,) to (5868,4)
		data_x = data[:, 3].astype(int)
		data_y = data[:, 2].astype(int)
		data_t = data[:, 1]
		data_p = data[:, 0].astype(int)

		# create a time window t_r to t_ru between 40% and 60% of the total time length
		t_min = np.min(data_t)
		t_max = np.max(data_t)

		ts = chunkify(x=number_of_chunks)
		all_chunks = [] # final output is appended to this

		for cnt, (r, s) in enumerate(zip(ts[0], ts[1])):
			t_r = t_min + ((t_max - t_min) * (r / 100))
			t_ru = t_min + ((t_max - t_min) * ((s) / 100))

			# index_need and index_need2 are the lower and upper indexes
			i_low = np.argmin(np.abs(data_t - t_r))
			i_up = np.argmin(np.abs(data_t - t_ru))

			x = []
			y = []
			p = []

			for index in range(i_low, i_up):
				x.append(data_x[index])
				y.append(data_y[index])
				p.append(data_p[index])


			# create a dataframe out of the three lists
			df1 = pd.DataFrame({'x': x, 'y': y, 'p': p}).sort_values(['x', 'y'])

			# divide data into two dataframes, for +ve and -ve values.
			df1_neg = df1[df1.iloc[:, 0] == -1]
			df1_pos = df1[df1.iloc[:, 0] == 1]

			# group by x and y, sum polarities for each x-y coordinate, and display x,y,p and summed p
			df1_neg['sum_p'] = df1_neg.groupby(['x', 'y'])['p'].transform(sum)
			df1_pos['sum_p'] = df1_pos.groupby(['x', 'y'])['p'].transform(sum)

			# drop old p, drop duplicates, and reset dataframe index
			df_neg = df1_neg.drop(['p'], axis=1).drop_duplicates(subset=['x', 'y', 'sum_p']).reset_index(drop=True)
			df_pos = df1_pos.drop(['p'], axis=1).drop_duplicates(subset=['x', 'y', 'sum_p']).reset_index(drop=True)

			# convert negative values to absolute
			df_neg['sum_p'] = df_neg['sum_p'].abs()

			# prepopulate 34x34x2 matrix with zeros (conversion to int32 for later use of fromfile)
			A = np.zeros(shape=(2, 100, 176), dtype=np.int32)

			# convert dataframe to np arrays B_neg and B_pos to allow easier indexing
			B_neg = df_neg.values
			B_pos = df_pos.values

			# B_pos -> A[1] and B_neg -> A[0]
			for row in B_neg:
				A[0][row[0]][row[1]] = row[2]
			for row in B_pos:
				A[1][row[0]][row[1]] = row[2]

			all_chunks.append(A)
		all_chunks = np.array(all_chunks)

		return all_chunks

def events_to_A(eventsPath,number_of_chunks):
	# load NVS text and converts to matrix

	data = np.loadtxt(eventsPath)

	# create lists for x, y, t, p columns
	data_x = data[:, 0].astype(int)
	data_y = data[:, 1].astype(int)
	data_t = data[:, 2]
	data_p = data[:, 3].astype(int)

	# create a time window t_r to t_ru between 40% and 60% of the total time length
	t_min = np.min(data_t)
	t_max = np.max(data_t)

	ts = chunkify(x=number_of_chunks)
	all_chunks = [] # final output is appended to this
	for cnt, (r, s) in enumerate(zip(ts[0], ts[1])):
		t_r = t_min + ((t_max - t_min) * (r / 100))
		t_ru = t_min + ((t_max - t_min) * ((s) / 100))

		# index_need and index_need2 are the lower and upper indexes
		i_low = np.argmin(np.abs(data_t - t_r))
		i_up = np.argmin(np.abs(data_t - t_ru))

		x, y, p = [],[],[]

		for index in range(i_low, i_up):
			x.append(data_x[index])
			y.append(data_y[index])
			p.append(data_p[index])

		# convert polarities that are zeroes to negative so that they can be stacked
		arr = np.array(p)
		arr[arr == 0] = -1
		p = list(arr)

		# create a dataframe out of the three lists
		df1 = pd.DataFrame({'x': x, 'y': y, 'p': p}).sort_values(['x', 'y'])

		# divide data into two dataframes, for +ve and -ve values.
		df1_neg = df1[df1.iloc[:, 0] == -1]
		df1_pos = df1[df1.iloc[:, 0] == 1]

		# group by x and y, sum polarities for each x-y coordinate, and display x,y,p and summed p
		df1_neg['sum_p'] = df1_neg.groupby(['x', 'y'])['p'].transform(sum)
		df1_pos['sum_p'] = df1_pos.groupby(['x', 'y'])['p'].transform(sum)

		# drop old p, drop duplicates, and reset dataframe index
		df_neg = df1_neg.drop(['p'], axis=1).drop_duplicates(subset=['x', 'y', 'sum_p']).reset_index(
			drop=True)
		df_pos = df1_pos.drop(['p'], axis=1).drop_duplicates(subset=['x', 'y', 'sum_p']).reset_index(
			drop=True)

		# convert negative values to absolute
		df_neg['sum_p'] = df_neg['sum_p'].abs()

		# prepopulate 34x34x2 matrix with zeros (conversion to int32 for later use of fromfile)
		A = np.zeros(shape=(2, 100, 176), dtype=np.int32)

		# convert dataframe to np arrays B_neg and B_pos to allow easier indexing
		B_neg = df_neg.values
		B_pos = df_pos.values

		# B_pos -> A[1] and B_neg -> A[0]
		for row in B_neg:
			A[0][row[1]][row[0]] = row[2]
		for row in B_pos:
			A[1][row[1]][row[0]] = row[2]

		all_chunks.append(A)

	all_chunks = np.array(all_chunks)
	return all_chunks


def run(tr_or_te,number_of_chunks):
	# saves each file as .npy after beheading. If no face found, use events text as normal
	global outputPath

	create_dir(tr_or_te)  # create train and test folders for beheaded set

	if(tr_or_te == 'tr'):
		videoPath_dir = videoPath_train
		eventsPath_dir = eventsPath_train
		outputPath = '{0}/n_Train_beheaded'.format(outputPath)
	elif(tr_or_te == 'te'):
		videoPath_dir = videoPath_test
		eventsPath_dir = eventsPath_test
		outputPath = '{0}/n_Test_beheaded'.format(outputPath)
	else:
		raise ValueError("Parameter 'tr_or_te' should be assigned as 'tr' for train set, and 'te' for test set")

	for class_index in classList:
		print("Looping over class: {0}".format(class_index))
		cnt = 0
		for (dirpath, dirnames, filenames) in os.walk('{0}'.format(eventsPath_dir)):
			for counter, name in enumerate(filenames):
				if class_index in name:
					cnt+=1
					print('Loading file: {}'.format(name))
					eventsPath = '{0}/{1}'.format(eventsPath_dir,name)
					videoPath = '{0}/{1}'.format(videoPath_dir,name[:-4]) # replace .mp4.txt with .mp4

					cnt += 1 # counter
					if (cnt % 100 == 0):
						print('{0} Files converted: {1}'.format(class_index,cnt))

					A = beheaded_to_A(videoPath,eventsPath,number_of_chunks)
					if A is None: # empty detection
						B = events_to_A(eventsPath,number_of_chunks)
						np.save('{0}/{1}/{2}'.format(outputPath, class_index, name[:-8]), B)
					else: # face detection
						np.save('{0}/{1}/{2}'.format(outputPath, class_index, name[:-8]), A)

for i in ['tr','te']:
	run(i,12)