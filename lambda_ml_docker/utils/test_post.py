# Copyright: (c) 2019, Bruno Calogero <brunocalogero@hotmail.com>
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)

"""
JSONify numopy array to be sent and send POST request via requests library.
"""
import numpy as np
import json
import requests

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def import_sequence_image(flag):
    """
    input: flag of wanted file import
    output: numpy array in desired
    """
    type = ''

    if flag == 0:
        data_path = '/Users/bcaloger/Desktop/lambda_ml_docker/utils/test_data/2999_Swiping_Down.npy'
        type = 'Swiping_Down'
    elif flag == 1:
        data_path = '/Users/bcaloger/Desktop/lambda_ml_docker/utils/test_data/691_Swiping_Up.npy'
        type = 'Swiping_Up'
    elif flag == 2:
        data_path = '/Users/bcaloger/Desktop/lambda_ml_docker/utils/test_data/2997_Swiping_Left.npy'
        type = 'Swiping_Left'
    elif flag == 3:
        data_path = '/Users/bcaloger/Desktop/lambda_ml_docker/utils/test_data/1945_Swiping_Right.npy'
        type = 'Swiping_Right'

    seq = np.load(data_path)
    seq = seq.reshape(1, 12, 100, 176, 2)

    # sanity check
    print('Class to be predicted: {0} ({1})'.format(flag, type))
    print('Image to be predicted of size:', seq.shape)

    return seq

def jsonify(data):
    """
    input: numpy data array containing time distributed timesteps of neuromorphioc frames
    output: jsonified numpy data array
    """

    json_dump = json.dumps({'body': data}, cls=NumpyEncoder)

    return json_dump


def post_request(data):
    """
    Handling Post Request
    """
    r = requests.post("https://831ygh4lc1.execute-api.eu-west-2.amazonaws.com/dev/handgestaction", data=data)

    return r


if __name__ == '__main__':

    # get numpy array from system
    data = import_sequence_image(3)
    # jsonify for POST request
    data = jsonify(data)
    # make POST request
    response = post_request(data)
    print('HTTP CALL RESULT', response.text) #TEXT/HTML
    print(response.status_code, response.reason) #HTTP
