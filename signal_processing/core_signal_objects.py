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
		
		self.st_window_size = st_window_size # Defaults are 50 m/s.
		self.st_window_stp = st_window_stp

	def _perform_short_term_feature_extraction(self):
		"""
		Method performs short term feature extraction on the signal
		based on the window and step of the amplitude timeseries
		"""
		# Calling pyAudioAnalysis to compute short term features:
		self.short_term_feature_matrix, self.short_term_feature_names = aF.feature_extraction(
			self.amplitude_timeseries, 
			self.sampling_rate,
			int(self.sampling_rate * self.st_window_size),
			int(self.sampling_rate * self.st_window_stp)
			)

	def _calculate_spectogram(self):
		"""
		Method that calculates a spectogram from the amplitude timeseries of 
		the audio signal based on the pyAudioAnalysis ShortTermFeatures methods. 
		"""
		# Calling pyAudioAnalysis to compute spectogram data matrix:
		self.specgram, self.time_axis, self.freq_axis = aF.spectrogram(
			self.amplitude_timeseries, 
			self.sampling_rate,
			int(self.sampling_rate * self.st_window_size),
			int(self.sampling_rate * self.st_window_stp),
			plot=True,
			show_progress=True
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
			self.amplitude_timeseries,
			n_fft=self.n_fft,
			hop_length=self.hop_length))

		self.stft_frequency = np.linespace(0, self.sampling_rate, 
			len(self.stft_magnitude))








