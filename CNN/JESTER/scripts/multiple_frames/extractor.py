# Copyright: (c) 2019, Bruno Calogero <brunocalogero@hotmail.com>
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import tensorflow as tf
import keras


# Setting up GPU / CPU, set log_device_placement to True to see what uses GPU and what uses CPU
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count = {'GPU': 1 , 'CPU': 1})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            base_model = InceptionV3(
                weights='imagenet',
                include_top=True
            )

            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )

        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, single_frame):
        # take a single frame from a single sequence and predict to get features
        single_frame = np.dstack((single_frame, np.zeros((100, 176)))) # !! Carefull here, had to expand to three dims for input !!
        x = np.expand_dims(single_frame, axis=0)
        # x = preprocess_input(x) # !! this standardizes the data in a special way, not the case for us, might have adverse effect !!

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features
