# ## Demo - Reconstructing a song using PCA

"""
Please note:

This content was copied from a Jupyter notebook i made - I had to convert it into a .py file since the
size of the .ipynb file was too big to be uploaded on GitHub (it was around 150 MB, which is way above
the 25 MB limit of GitHub).

I didn't use scikit-learn's PCA implementation, since I wanted to practice for my upcoming Data Science
exam by implementing PCA from scratch using SVD, and also because i wanted to have more control over the
components we keep (for example, i wanted to try keeping the middle components instead of the top ones,
which I don't think is possible with scikit-learn's PCA since it only keeps the top components.

Also, I can't upload the original .mp3 file of the song i used for this demo, since it is copyrighted
material, so you will have to use your own .mp3 file and change the filename variable accordingly
(and make sure to convert it into a .wav file using ffmpeg, as explained in the code).

The song I used was "Everything In Its Right Place" by Radiohead, one of the best songs ever made,
from one of the best albums ever made, from one of the best bands ever existed.
"""

import subprocess
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from IPython.lib.display import Audio

# Loading the audio file

basefolder = "/Users/fp/GitHub Repos/UNISA-Vault/MastersDegree/Data Science/Exercises/Python scripts/"
filename = "song.mp3"
target = basefolder + "song.wav"

# we need to convert this mp3 file into a wav file, so we can load it as a numpy array
# i am going to use ffmpeg (make sure you have it installed on your system)

subprocess.run(["rm", target])  # remove the target file if it already exists
subprocess.run(["ffmpeg", "-i", basefolder + filename, target])

# Let's convert te audio into a numpy array and let's see what we get

original_song = Audio(target)

# load the wav file as a numpy array
sample_rate, data = wavfile.read(target)

print("Sample rate:", sample_rate)    # number of samples per second
print("Data shape:", data.shape)      # (number of samples, number of channels)
print("Data type:", data.dtype)

X_left = data[:, 0]   # left channel
X_right = data[:, 1]  # right channel

# Let's plot the left and right channels
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(X_left)
plt.title("Original - Left Channel")
plt.subplot(1, 2, 2)
plt.plot(X_right)
plt.title("Original - Right Channel")
plt.show()

original_song

# since we have two channels, let's work on the two channels separately and then we will combine them again at the end

# let's define the frame length (number of samples per frame) and split the audio into frames
frame_len = 1024
n_frames = len(X_left) // frame_len

X_left_frames = X_left[:n_frames * frame_len].reshape(n_frames, frame_len).astype(data.dtype)
X_right_frames = X_right[:n_frames * frame_len].reshape(n_frames, frame_len).astype(data.dtype)

print("Left channel frames shape:", X_left_frames.shape)   # (number of frames, frame length)
print("Right channel frames shape:", X_right_frames.shape) # (number of frames, frame length)

mu_left = np.mean(X_left_frames, axis=0)
mu_right = np.mean(X_right_frames, axis=0)

U_left, S_left, Vt_left = np.linalg.svd(X_left_frames - mu_left, full_matrices=False)
U_right, S_right, Vt_right = np.linalg.svd(X_right_frames - mu_right, full_matrices=False)

# let's plot the singular values to see how many components we need to keep
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(S_left)
plt.title("Singular values - Left Channel")
plt.subplot(1, 2, 2)
plt.plot(S_right)
plt.title("Singular values - Right Channel")
plt.show()

# let's keep the top 100 components (out of 1024) and reconstruct the audio
k = 100   # the less the components, the more we are compressing the audio, hence the more we are losing information,
          # ending up with a pretty bad approximation of the original audio. but the more we are compressing it,
          # the more we are reducing the size of the audio file, so it's a trade-off between quality and size.

X_left_reduced = mu_left + (U_left[:, :k] @ np.diag(S_left[:k]) @ Vt_left[:k, :])
X_right_reduced = mu_right + (U_right[:, :k] @ np.diag(S_right[:k]) @ Vt_right[:k, :] )

# let's plot the approximated left and right channels
plt.figure(figsize=(15, 5))
plt.subplot(2, 2, 1)
plt.plot(X_left.flatten())
plt.title("Original - Left Channel")
plt.subplot(2, 2, 2)
plt.plot(X_left_reduced.flatten())
plt.title("Approximated - Left Channel")
plt.subplot(2, 2, 3)
plt.plot(X_right.flatten())
plt.title("Original - Right Channel")
plt.subplot(2, 2, 4)
plt.plot(X_right_reduced.flatten())
plt.title("Approximated - Right Channel")
plt.tight_layout
plt.show()

# let's combine the left and right channels back into a stereo audio
data_approx_left = X_left_reduced.flatten()
data_approx_right = X_right_reduced.flatten()

data_approx = np.stack((data_approx_left, data_approx_right), axis=-1)

# now let's listen to it
# NOTE: we give the transposed data to the Audio function because it expects the shape to be (number of channels, number of samples)
approximated_song = Audio(data_approx.T, rate=sample_rate)
approximated_song

# What if i don't pick the top 50 components, but the middle ones? Let's say from 50 to 100:

k1 = 50
k2 = 100

X_left_reduced = mu_left + (U_left[:, k1:k2] @ np.diag(S_left[k1:k2]) @ Vt_left[k1:k2, :])
X_right_reduced = mu_right + (U_right[:, k1:k2] @ np.diag(S_right[k1:k2]) @ Vt_right[k1:k2, :] )

data_approx_left = X_left_reduced.flatten()
data_approx_right = X_right_reduced.flatten()

data_approx = np.stack((data_approx_left, data_approx_right), axis=-1)

approximated_song = Audio(data_approx.T, rate=sample_rate)
approximated_song

# The result (as we would totally expect) is an almost totally incomprehensible audio - everything is just distorted and glitchy (and definitely not in its right place - ba-dum tss...)
