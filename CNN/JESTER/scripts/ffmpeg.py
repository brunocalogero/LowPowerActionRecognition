import os
from csv_parse import parse_swiping
swipe_l, swipe_r, swipe_u, swipe_d = parse_swiping()

#JESTER preprocessing using ffmpeg
#convert folder of jpegs into mp4s for each of the 4 labels

for item in swipe_l:
	os.system("ffmpeg -f image2 -r 37 -i C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/20bn-jester-v1/{}/%5d.jpg D:/PIX2NVS/input/{};Swiping_Left.mp4".format(item,item))
for item in swipe_r:
	os.system("ffmpeg -f image2 -r 37 -i C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/20bn-jester-v1/{}/%5d.jpg D:/PIX2NVS/input/{};Swiping_Right.mp4".format(item,item))
for item in swipe_u:
	os.system("ffmpeg -f image2 -r 37 -i C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/20bn-jester-v1/{}/%5d.jpg D:/PIX2NVS/input/{};Swiping_Up.mp4".format(item,item))
for item in swipe_d:
	os.system("ffmpeg -f image2 -r 37 -i C:/Users/millir/Desktop/4thyearproject/LowPowerActionRecognition/CNN/datasets/JESTER/20bn-jester-v1/{}/%5d.jpg D:/PIX2NVS/input/{};Swiping_Down.mp4".format(item,item))

'''
print(len(swipe_l))
print(len(swipe_r))
print(len(swipe_u))
print(len(swipe_d))
'''