# Signal-Reconstruction-with-Tensorflow
Reconstruction of a 1D signal from it's magnitude spectrogram using tensorflow

Simple example of using tensorflow to reconstruct a signal from its magnitude spectrogram. Also runs the Griffin-Lim algorithm on the same magnitude spectrogram. The resulting signals are perceptually indistinguishable, though both sub-optimal. Basically shows that you can reconstruct the signal very easily in tensorflow and in much less time than running Griffin-Lim.

Just run signal_reconstruction.py, command line arguments:
* --audio_file : path to audio file to use, otherwise uses default
* --output : path to output folder, otherwise doesn't output anything
* --extras : boolean, if true, saves reconstructions with random and zero phase

Default sample is taken from the Common Voice Dataset (cv-other-test folder)
Griffin-Lim Implementation is taken from : https://github.com/tensorflow/magenta/blob/master/magenta/models/nsynth/utils.py
