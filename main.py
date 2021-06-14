import csv
import os
import pathlib
from pydub import AudioSegment
import progressbar

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from IPython import display
from IPython.display import Audio

files = {}
path = "C:\\Users\\Vinetos\\Desktop\\cv-corpus-6.1-2020-12-11\\fr"
# Read validated input data
with open(path + '\\validated-trans.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)  # skip header
    for row in reader:
        files[row[1][:-4]] = row[2]  # filename: sentence

print("Converting to wav")
# Convert the dataset to wav files
with progressbar.ProgressBar(max_value=len(files
                                           )) as bar:
    i = 0
    for filename in files:
        # convert mp3 towav
        conv = 'converted/' + filename + '.wav'
        if not os.path.isfile(conv):
            sound = AudioSegment.from_mp3(path + "\\clips\\" + filename + ".mp3")
            sound.export(conv, format="wav")

        bar.update(i + 1)
        i += 1

print("Converting to FFT png")
# Convert the dataset to wav files
with progressbar.ProgressBar(max_value=len(files
                                           )) as bar:
    plt.figure(figsize=(5, 3))
    i = 0
    for filename in files:
        # convert wav to mp3
        audio = tfio.audio.AudioIOTensor("converted/" + filename + ".wav")

        # remove last dimension
        audio_tensor = tf.squeeze(audio[:], axis=[-1])
        Audio(audio_tensor.numpy(), rate=audio.rate.numpy())
        tensor = tf.cast(audio_tensor, tf.float32) / 32768.0

        # Convert to spectrogram
        spectrogram = tfio.audio.spectrogram(
            tensor, nfft=512, window=512, stride=256)

        # Convert to mel-spectrogram
        mel_spectrogram = tfio.audio.melscale(
            spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)

        # Convert to db scale mel-spectrogram
        dbscale_mel_spectrogram = tfio.audio.dbscale(
            mel_spectrogram, top_db=80)

        plt.clf()
        plt.imshow(dbscale_mel_spectrogram.numpy().transpose())
        plt.axis('off')
        plt.savefig("fft/" + filename + ".png")
        bar.update(i + 1)
        i += 1