This will hold the documentation for all research made on auto-encoders, use it as a means to keep track of the work you have done
and, eventually as something that allows people to understand what you have been up to.

Use this to also keep track of all the resources that you have used to come up with your solution.

As a starting point you can play with the MNIST DVS data set / or the normal MNIST and create your small autoencoder:
https://www.jeremyjordan.me/autoencoders/ (this is a good starting point)

I would also explore and gather the best possible understanding as to where this auto-encoder / compressor/decompressor
will be integrated in the overall system, benefits of using one over the other and most importantly, benchmarks.
Don't be scared to send personal emails to Yiannis asking questions and what not to gather a better understanding of what the expectations are here.


Always keep in mind - the title of our project is Low Power Action recognition - keep in mind how compression is helping achieve the objective of the project, (in this case auto-encoding would mimic the easier file transfer from the DVS camera to the portable system in an efficient way). Everytime you see that something from this autoencoder can help achieve our objectives right it down and pivot towards that objective.

Version control for autoencoder:
ver1 - MNIST "VAE.ipynp"
ver2 - NMNIST "VAE-NMNIST.ipynb" -> first nmnist test with huge clump of data
ver3 - VAE-NMNIST.ipynb" ->Trying different layers, neurons, activation function, epochs. Result in a better seperation of data
ver4 - "Justin_test2 & Chingis_test"-> Different pre-processing method to remove background noise, added plot
ver5 - "Justin_test3"-> Hyperameterisation, finalise the model with key metrics
ver6 - "SVM_etc" -> Trying to compare with SVM for VAE_data and Normal_data
ver7 - "CVAE" -> First try implementation of CVAE

References:
https://blog.keras.io/building-autoencoders-in-keras.html  -  Building autoencoders in Keras
https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
https://github.com/llSourcell/autoencoder_explained/blob/master/variational_autoencoder.py
https://github.com/tawheeler/2017_iv_deep_radar/blob/master/VAE.py   -    examples of the VAE
http://deeplearning.buzz/2017/06/01/what-is-batch-size-and-epoch-in-neural-network/   -   batch sizes and epochs explained
https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f    -    Deep inside: Autoencoders
https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798    -     Applied Deep Learning - Part 3: Autoencoders

https://gombru.github.io/2018/05/23/cross_entropy_loss/ - binary-cross-entropy
https://tdhopper.com/blog/cross-entropy-and-kl-divergence/ - KL divergeance and cross-entropy

CVAE using keras :https://gist.github.com/naotokui/b9fb93b8dba534b55a140e8c88ce07f5
