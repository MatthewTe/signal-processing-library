# Importing Audio libraries:
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import ShortTermFeatures as aF
import librosa

# Importing external packages:
import numpy as np
import pandas as pd
import math
from scipy.fft import fft, ifft
import random 
import matplotlib.pyplot as plt

class AudioSignal(object):
	"""
	Object representing an audio signal. Serves as a wrapper for pyAudio and 
	librosa methods.
	"""
	def __init__(self, audio_path, st_window_size=0.050, st_window_stp=0.050):
		# Declaring Instance parameters:
		self.amplitude_timeseries, self.sampling_rate = librosa.load(audio_path)
		self.signal_duration = len(self.amplitude_timeseries / float(self.sampling_rate))

		# Cleaing the audio signal of leading and trailing scilence:
		self.cleaned_amplitude, _ = librosa.effects.trim(self.amplitude_timeseries)
		self.cleaned_signal_duration = len(self.cleaned_amplitude) / float(self.sampling_rate)
		
		self.st_window_size = st_window_size # Defaults are 50 m/s.
		self.st_window_stp = st_window_stp

	def _perform_short_term_feature_extraction(self):
		"""
		Method performs short term feature extraction on the signal
		based on the window and step of the amplitude timeseries
		"""
		# Calling pyAudioAnalysis to compute short term features:
		self.short_term_feature_matrix, self.short_term_feature_names = aF.feature_extraction(
			self.cleaned_amplitude, 
			self.sampling_rate,
			int(self.sampling_rate * self.st_window_size),
			int(self.sampling_rate * self.st_window_stp)
			)

	def _calculate_short_term_fourier_transform(self, n_fft, hop_length=512):
		"""
		Method uses the librosa library to calculate tge Short Term Fourier Transform. All of
		the STFT data is calculated via the librosa.stft() method. 
		"""
		# Declaring the FFT window size and STFT audio frame length as instance params:
		self.n_fft, self.hop_length = n_fft, hop_length

		# Performing the librosa Short Term Fourier Transform calculations:
		self.stft_magnitude = np.abs(librosa.stft(
			self.cleaned_amplitude,
			n_fft=self.n_fft,
			hop_length=self.hop_length))

		self.stft_frequency = np.linspace(0, self.sampling_rate, 
			len(self.stft_magnitude))

	def _calculate_spectogram(self):
		"""
		The method that performs data transformations of STFT data generated from the
		self._calculate_short_term_fourier_transform() methods. The data transformations
		applied on spectogram data are:

		* STFT amplitude spectogram --> Decible Scaled Spectogram
		* A Mel Spectogram 
		* A Mel Spectogram --> Mel Decible Scaled Spectogram
		
		"""
		# Converting the STFT magnitude (amplitude spectogram) to Decibel scale:
		self.stft_Db_specdata = librosa.amplitude_to_db(
			self.stft_magnitude, ref=np.max)

		# Calculating the Mel Spectograms:
		self.mel_specdata = librosa.feature.melspectrogram(
			self.cleaned_amplitude, sr=self.sampling_rate)

		# Converting the Mel Spectogram dat to a decible scale: 
		self.mel_Db_specdata = librosa.amplitude_to_db(
			self.mel_specdata, ref=self.sampling_rate)

	def calculate_mel_frequency_cepstral_coeffs(self, num_mfcc=13):
		"""
		The method makes use of the librosa library to compute the Mel Frequency
		Cepstral Coefficients (MFCCs). 
		"""
		# Creating instance parameters:
		self.num_mfcc = num_mfcc

		# Calculating the MFCC from the cleaned amplitude timeseries:
		self.mfccs = librosa.feature.mfcc(
			self.cleaned_amplitude, 
			n_fft= self.n_fft,
			hop_length= self.hop_length,
			n_mfcc=num_mfcc)


