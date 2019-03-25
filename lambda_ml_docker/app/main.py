# Copyright: (c) 2019, Bruno Calogero <brunocalogero@hotmail.com>
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)

import json
from io import BytesIO

try:
    import unzip_requirements
except ImportError:
    pass

import tensorflow as tf
import numpy
import boto3

def get_model():
    # Access model from S3 using boto3
    bucket = boto3.resource("s3").Bucket("modelhoster")
    with BytesIO() as modelfo:
        # Download file into file object that we created above
        bucket.download_fileobj(Key='model/lrcn_model.hdf5', Fileobj=modelfo)
        # Deserialize the model
        model = tf.keras.models.load_model(modelfo)
    return model

def predict(event):
    json_data = json.loads(event['body'])
    body = json_data["body"]
    data = numpy.asarray(body)
    model = get_model()
    result = model.predict(data)
    return result.tolist()

def lambda_handler(event, context):
    result = predict(event)
    result = str(result)
    return {"statusCode": 200,
            "body": json.dumps(result)}
