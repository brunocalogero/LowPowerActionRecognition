'''
This file accepts the address for Master as input argument

Example usage:

    `python script.py 192.168.1.4`

    This will replace all the current mainAddr values to the user desired Master Address

'''

import re
import sys

assert (len(sys.argv) > 1), "Please input address of Master as a string"

input_address = sys.argv[1]

print "setting mainAddr to {0}".format(input_address)

new_address = 'mainAddr = {0}'.format(input_address)

# Read in the file
with open('config.properties', 'r') as file:
    filedata = file.read()

# Replace the target string
new_text = re.sub(r"mainAddr = .*", new_address, filedata)

# Write the file out again
with open('config.properties', 'w') as file:
    file.write(new_text)
