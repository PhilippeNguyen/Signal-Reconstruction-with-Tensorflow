import tensorflow as tf
import argparse
import librosa
import numpy as np
from distutils.util import strtobool

import os
fs = os.path.sep

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--audio_file', action='store',
                    dest='audio_file',default=None,
                    help='path to the audio_file')
parser.add_argument('--output', action='store',
                dest='output',
                default=None,
                help='name of output folder')
parser.add_argument('--extras', action='store',
                dest='extras',
                default=True,type=strtobool,
                help='Boolean, whether to save reconstruction with 0 phase and random phase')
args = parser.parse_args()
if args.output is not None:
    output = args.output if args.output.endswith(fs) else args.output + fs
    os.makedirs(output,exist_ok=True)
else:
    output = None
sess = tf.Session()

N_FFT = 2048
hop_length = N_FFT//4
if args.audio_file is None:
    x, sr = librosa.load(os.getcwd()+fs+'sample-001187.mp3')
else:
    x, sr = librosa.load(args.audio_file)

#fix signal to be multiple of hops (to match stft)
n_hops = len(x)//hop_length
x = x[:(n_hops*hop_length)]

#stft of input
S = tf.contrib.signal.stft(x,frame_length=N_FFT,frame_step=hop_length).eval(session=sess)

#magnitude spectrogram
mag = np.abs(S)

#%%Signal Reconstruction with TF section, using L-BFGS
alpha = 100.0 #You can set this to 0 to see what it's like to optimize without regard to phase
num_tf_iter = 300

init_recon = np.random.randn(*x.shape).astype(np.float32)
recon = tf.Variable(init_recon)
sess.run(tf.global_variables_initializer())

stft_tf = tf.contrib.signal.stft(recon,frame_length=N_FFT,frame_step=hop_length)
mag_tf = tf.abs(stft_tf)
x_tf = tf.contrib.signal.inverse_stft(stft_tf,frame_length=N_FFT,frame_step=hop_length)

x_loss = tf.reduce_sum(tf.square(recon-x_tf))
mag_loss = tf.reduce_sum(tf.square(mag-mag_tf))

loss = alpha*x_loss + mag_loss

opt = tf.contrib.opt.ScipyOptimizerInterface(
      loss, method='L-BFGS-B', options={'maxiter': num_tf_iter},
      var_list=[recon])
# Optimization
print("Optimizing")
opt.minimize(sess)
print("Final Loss: ", loss.eval(session=sess))
tf_recon = recon.eval(session=sess)

if output is not None:
    librosa.output.write_wav(output+'tf_recon.wav',tf_recon,sr=sr)
    
#%% griffin lim section
num_gl_iters=300

#Calculate the stft and magnitude spectrogram of the original signal using librosa
#Note: the lengths of the stft using librosa are slightly longer than using tensorflow's version
S_lib = librosa.core.stft(x,n_fft=N_FFT,hop_length=hop_length)
mag_lib = np.abs(S_lib)

#Run the Griffin Lim alg.
# Obtained : https://github.com/tensorflow/magenta/blob/master/magenta/models/nsynth/utils.py
print("Running Griffin-Lim")
phase_angle = 2 * np.pi * np.random.random_sample(mag_lib.shape) - np.pi
def inv_magphase(mag, phase_angle):
  phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
  return mag * phase

complex_specgram = inv_magphase(mag_lib, phase_angle)
for i in range(num_gl_iters):
    audio = librosa.istft(complex_specgram,win_length=N_FFT,hop_length=hop_length)
    if i != num_gl_iters - 1:
        complex_specgram = librosa.stft(audio,n_fft=N_FFT,hop_length=hop_length)
        _, phase = librosa.magphase(complex_specgram)
        phase_angle = np.angle(phase)
        complex_specgram = inv_magphase(mag_lib, phase_angle)

# Compute loss, do this by using the tensorflow loss 
sess.run(recon.assign(audio))
gl_loss = loss.eval(session=sess)
print("Final Griffin-Lim Loss",gl_loss)
if output is not None:
    librosa.output.write_wav(output+'gl_recon.wav',audio,sr=sr)
    
    #Save Extras
    if args.extras:
        rand_phase_angle = 2 * np.pi * np.random.random_sample(mag_lib.shape) - np.pi
        rand_complex_specgram = inv_magphase(mag_lib, rand_phase_angle)
        rand_audio = librosa.istft(rand_complex_specgram,win_length=N_FFT,hop_length=hop_length)
        librosa.output.write_wav(output+'rand_phase.wav',rand_audio,sr=sr)
        
        zerop_complex_specgram = inv_magphase(mag_lib+np.pi, np.zeros_like(phase_angle))
        zerop_audio = librosa.istft(zerop_complex_specgram,win_length=N_FFT,hop_length=hop_length)
        librosa.output.write_wav(output+'zero_phase.wav',zerop_audio,sr=sr)
