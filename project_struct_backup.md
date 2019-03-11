## WRITTEN BY: BRUNO CALOGERO

THANKS
* insert all the people we would like to thank (not to forget: Yiannis + Yin Bi)
* Daniel Schep (Serverless )


INTRODUCTION (Bruno)
* background
    * DVS cameras 101(Mihir)
    * machine learning history 101 (use article/website given to me by Mark Sullivan - also goods inspo from some of the research papers in git md file)
* overall idea of what we are trying to achieve
* what this will mean in terms of learning (new technical skills and processes)
    * had to take cs231n class to learn about everything from linear classification to more complex CNN all the way up to LSTMs (learnt tensorflow(keras) and pytorch from scratch, plus the usual data processing libraries like pandas and PIL but also other ML related libraries such as sklearn...)
    * had to learn docker, unit-testing, proper code packaging/structure -> makefile, pipfile, npm dependencies, setup.py/setup.cfg, bash scripts
    * automated deployment: serverless (multi-vendor friendly framework on top AWS APIGatgeway in our case)
    * AWS: S3, Lambda, APIGateway


PREAMBLE:
*  NMNIST DATA and Preprocessing (Mihir)
    * conversion of data tuples into particular images capturing temporal data for a given delta T
    * 34x34x1
    * 34x34x2 standard
    * 34x34x2 interleaved
* models tested
    * KNN (Justin)
    * SVM (Bruno)
    * CNN (different structures) (Bruno)
* VAE initial work (Justin & Chingis)
* Initial tests on frontier (Bruno)
* initial tests on AWS (Bruno)


CLOUD and EDGE based action-recognition framework:
* NJESTER the dataset (Bruno)
    * (relevant subsections)
    * large datasets and problems that come with them (resort to having to use batching methodologies)
    * GPU setup and playing key role in processing (Keras and Cudnn)
* Necessary Pre-processing (Mihir)
    * (other relevant subsections)
* ML Framework (Bruno)
    * structure
        * models
        * extractor
        * data importer
        * importance of generators
            * avoid loading huge datasets in
        * importance of threading
    * Explored Architectures (for each also  )
        * InceptionV3 + LSTM (chunks of data, w/ and w/o face noise removal)
            * effects and results of normalising, small amount of classes, large amount of classes, benchmark vs RGB dataset
        * InceptionV3 + MLP (chunks of data, w/ and w/o face noise removal)
        * LRCN (Time-distributed CNN + LSTM) (chunks of data, w/ and w/o face noise removal)
        * C3D (custom tiny version)
            * problem with initial architecture
        * transfer learning with InceptionV3 (single frame) (more useful with more classes)
    * Comparative Conclusions and Observations
        * consider including what will actually be used in practice for the Demo
* Encoding/Decoding (Chingis & Justin)
    * necessity of encoding data for efficient and fast transfer (for later predictions)
    * VAEs (why this and not something else create subsections for further discussion)
    * CVAEs
    * (other relevant subsections)
* Cloud-based approach (whole setup) (Bruno):
    * diagram of the full system: AWS lambda + S3 + SQS + API Gateway + Serverless
    * Lambda limits and file sizes (need to take into consideration before deployment)
    * aws-cli
        * sharing IAM profile
        * creating a bucket to store our model (bucket name must be unique, whatever location constraints, if bucket is private or public, lambda must be in the same region / flaskapp
        * send model to bucket: aws s3 cp model/pipeline.pkl s3://modelhoster/model/model.pkl
    * dockerized project + Serverless with pipfile dependencies
    * taking into account limitations:
        * zip true with import
        * increase timeout to 30s
        * 250 MB limitations
    * Cloudwatch logs for debugging
* Edge-based Approach (Bruno):
    * Frontier-based app (payload and processing distributed over multiple raspberry-pis)



CONCLUSION:
* What we have learnt
* Future work
    * using SQS for unlimited asynchronous requests to lambdas
    * separate lambda function for decryption (layers as well for individual requirement?)
    * added database to avoid API Gateway 30s limit timeout
* future of this tech and were it might lead us with current industry


saving the model in tensor flow: https://stackoverflow.com/questions/51678609/saving-trained-tensorflow-model-to-inference-on-another-machine?noredirect=1&lq=1
https://stackoverflow.com/questions/36281129/no-variable-to-save-error-in-tensorflow
https://askubuntu.com/questions/420981/how-do-i-save-terminal-output-to-a-file
https://stackoverflow.com/questions/38626435/tensorflow-valueerror-no-variables-to-save-from
https://www.tensorflow.org/serving/serving_basic
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md
