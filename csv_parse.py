import pandas as pd

train_file = 'C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/jester-v1-train.csv'
test_file = 'C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/jester-v1-validation.csv'

#create lists for 4 'Swiping' labels

def parse_swiping(train_file, test_file):

	d = pd.read_csv('{0}'.format(train_file)).values #resulting in 16725 videos for 4 labels
	#store indexes for the label
	swipe_l_tr = [item[0].split(';')[0] for item in d if "Swiping Left" in item[0]]
	swipe_r_tr = [item[0].split(';')[0] for item in d if "Swiping Right" in item[0]]
	swipe_u_tr = [item[0].split(';')[0] for item in d if "Swiping Up" in item[0]]
	swipe_d_tr = [item[0].split(';')[0] for item in d if "Swiping Down" in item[0]]

	e = pd.read_csv('{0}'.format(test_file)).values  # resulting in 2008 videos for 4 labels
	# store indexes for the label
	swipe_l_te = [item[0].split(';')[0] for item in e if "Swiping Left" in item[0]]
	swipe_r_te = [item[0].split(';')[0] for item in e if "Swiping Right" in item[0]]
	swipe_u_te = [item[0].split(';')[0] for item in e if "Swiping Up" in item[0]]
	swipe_d_te = [item[0].split(';')[0] for item in e if "Swiping Down" in item[0]]

	return swipe_l_tr, swipe_r_tr, swipe_u_tr, swipe_d_tr, swipe_l_te, swipe_r_te, swipe_u_te, swipe_d_te