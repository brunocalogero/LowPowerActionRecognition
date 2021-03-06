# WRITTEN BY: BRUNO CALOGERO

- https://arxiv.org/abs/1707.04555 -
Temporal Modeling Approaches for Large-scale Youtube-8M Video Understanding (fast-forward LSTMs for video prediction generation)
- https://arxiv.org/abs/1609.08675 - YouTube-8M: A Large-Scale Video Classification Benchmark
- https://machinelearningmastery.com/when-to-use-mlp-cnn-and-rnn-neural-networks/ - different scenarios of when to use what architecture
- https://machinelearningmastery.com/models-sequence-prediction-recurrent-neural-networks/ - slightly more details on RNNs and use cases for prediction generation
- https://machinelearningmastery.com/cnn-long-short-term-memory-networks/ - implementation details to get started (with keras)
- https://arxiv.org/abs/1411.4389 - related with previous link (one of original papers for CNN+LSTMs for vision description problems, very good for benchmarking)
- https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/ - other practical keras based implementation
- https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43455.pdf - another paper for benchmarking (adds a DNN after CNN+LSTM)
- https://adventuresinmachinelearning.com/keras-lstm-tutorial/ - tutorial on LSTMs in Keras
- https://jhui.github.io/2017/03/15/RNN-LSTM-GRU/ - LSTM vs RNN vs GRU
- https://www.youtube.com/watch?v=4tlrXYBt50s - RNNs vs LSTMs vs GRUs (great video summary of the differences, advantages and mathematical understanding)
- Lecture 10, 12 (and 13) of CS231n, especially 10 for RNNs and LSTMs and 12 for Variational Autoencoders
- https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5 - CNN + LSTM implementation
- https://github.com/sagarvegad/Video-Classification-CNN-and-LSTM-/blob/master/train_CNN_RNN.py - same as above
- https://github.com/HHTseng/video-classification - 3DCNN and CNN+LSTM implementations
- https://github.com/woodfrog/ActionRecognition - action recognition with pixel data (w/ optical flow also CNN + LSTM)
- https://github.com/talhasaruhan/video-action-classification - same as above
- https://riptutorial.com/keras/example/29812/vgg-16-cnn-and-lstm-for-video-classification - simple same as above
- https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5 - same as above nice progression from shitty models to good one


Use the youtube 8M part 2 paper, great way to describe more complex approaches (SOA) compared to our more basic one, possibly implement??
https://static.googleusercontent.com/media/research.google.com/en//youtube8m/workshop2018/c_01.pdf
--> main architecture used: Willow
--> temporal aggregation (aggregate frame-level features) (from frames to video features): NetVLAD (basic LSTM/GRU approaches were also used, bidirectional LSTMs also used by Samsung)
--> Convolution on temporal axis is another famous method to substitute RNN based methods (ResidualCNN-X: where X is the output size, composed of a fully connected layer and a deep CNN network with several residual modules.
(time-distributed convolutional layers, containing several layers of convolutions followed by max-pooling for video and audio separately, then concatenating the resulting features.)

- https://static.googleusercontent.com/media/research.google.com/en//youtube8m/workshop2018/c_17.pdf --> good fast explanation of used LSTMs + ResidualCNN-X explanation
- https://arxiv.org/pdf/1409.2329.pdf - for RNN explanation
- https://arxiv.org/pdf/1411.4389.pdf - main paper explaining the architecture we are trying to achieve
(First, we integrate 2D CNNs that can be pre-trained on large image datasets. Second, we combine the CNN and LSTM into a single model to enable end-to-end fine-tuning)
- http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review - really good for video action recognition basic approaches (all the different techniques until now)



- http://crcv.ucf.edu/data/UCF101.php - UCF101 dataset where first tests are made for CNN + LSTM
citation: Khurram Soomro, Amir Roshan Zamir and Mubarak Shah, UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild., CRCV-TR-12-01, November, 2012.
- https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5 - implementations with CNN + LSTM and actual coding help with github blog below
- https://github.com/harvitronix/five-video-classification-methods



- https://arxiv.org/pdf/1804.08150.pdf - Recurrent Spiking Neural Networks and a bunch of examples of what is currently out there for SNN usage and how they have been used
- https://arxiv.org/pdf/1610.09513.pdf - Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences (for future)

- https://arxiv.org/abs/1512.00567 -  Inception V3, pre-trained on ImageNet.


- https://blog.coast.ai/continuous-online-video-classification-with-tensorflow-inception-and-a-raspberry-pi-785c8b1e13e1 (about transfer learning, offline learning and retraining last layers of Inceoption + continuous online classification)
- https://blog.coast.ai/continuous-video-classification-with-tensorflow-inception-and-recurrent-nets-250ba9ff6b85
