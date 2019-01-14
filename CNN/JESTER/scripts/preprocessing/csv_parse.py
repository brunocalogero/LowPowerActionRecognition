import pandas as pd

filename = 'C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/jester-v1-train.csv'


#create lists for 4 'Swiping' labels

def parse_swiping(filename):

	d = pd.read_csv('{0}'.format(filename), nrows=7500).values #resulting in approximately 1000 videos for 4 labels

	#store indexes for the label
	swipe_l = [item[0].split(';')[0] for item in d if "Swiping Left" in item[0]]
	swipe_r = [item[0].split(';')[0] for item in d if "Swiping Right" in item[0]]
	swipe_u = [item[0].split(';')[0] for item in d if "Swiping Up" in item[0]]
	swipe_d = [item[0].split(';')[0] for item in d if "Swiping Down" in item[0]]

	return swipe_l, swipe_r, swipe_u, swipe_d