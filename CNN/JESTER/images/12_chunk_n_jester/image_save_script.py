import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image

# set to 0 for positive polarity and 1 for negative polarity (or vice-versa)
pol = 0
# change to preffered name for file when saved
name_file = '140946_swipe_right'

for i in range(0,11):
    img = data[i]
    img = img[0]
    image.imsave('{0}_{1}.png'.format(name_file, i), img)
