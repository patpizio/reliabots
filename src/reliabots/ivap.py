import numpy as np
from sklearn.metrics import classification_report
from sklearn.isotonic import IsotonicRegression
from .VennABERS import ScoresToMultiProbs
from .calibrutils import reliabubble, calibration_errors
from scipy.special import softmax


class IVAP:

	def __init__(self, y_cal, y_test, y_cal_pred, y_test_pred, label=1, apply_softmax=False):
		self.y_cal = y_cal
		self.y_test = y_test
		self.y_cal_pred = y_cal_pred
		self.y_test_pred = y_test_pred
		self.label = label
		if apply_softmax == True:
			self.y_cal_pred = softmax(self.y_cal_pred, axis=1)
			self.y_test_pred = softmax(self.y_test_pred, axis=1)
		self.y_cal_scores = self.y_cal_pred[:, self.label]
		self.y_test_scores = self.y_test_pred[:, self.label]
		self.p_0, self.p_1 = self.__get_multiprobs()
		self.p_single = self.__compute_single_prob()


	def __get_multiprobs(self):
		calibr_points = list(zip(self.y_cal_scores, self.y_cal))
		return ScoresToMultiProbs(calibr_points, self.y_test_scores)

	def __compute_single_prob(self):
		return self.p_1 / (1 - self.p_0 + self.p_1)

	def prob_uncertainty(self):
		return np.abs(self.p_0 - self.p_1)

	def print_performance(self, digits=2):
		y_hat_test = np.round(self.p_single).astype(int)
		print(classification_report(self.y_test, y_hat_test, digits=digits))

	def compute_calibration_errors(self, num_bins=10):
		return calibration_errors(self.p_single, self.y_test, num_bins)

	def plot_reliabubble(self, num_bins=10, size_max=20, font_size=12, title=None):
		return reliabubble(self.p_single, self.y_test, num_bins, size_max, font_size, title=title)

	def compare_with_isotonic_regression(self, num_bins=10):
		iso_reg = IsotonicRegression(out_of_bounds='clip').fit(self.y_cal_scores, self.y_cal)
		y_test_iso = iso_reg.predict(self.y_test_scores)
		print('Comparison with direct isotonic regression:')
		print(calibration_errors(y_test_iso, self.y_test, num_bins, include_log_loss=False))
		return reliabubble(y_test_iso, self.y_test, num_bins, title='Isotonic Regression')


			