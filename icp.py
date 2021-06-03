import numpy as np
from sklearn.metrics import classification_report
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ConformalPredictor():

	def __init__(self, y_cal_proba, y_cal, y_test_proba, y_test, labels, mondrian=True):
		self.y_cal_proba = y_cal_proba
		self.y_cal = y_cal
		self.y_test_proba = y_test_proba
		self.y_test = y_test
		self.labels = labels
		self.alphas = np.zeros(len(y_cal))  # nonconformity scores for calibration set
		self.ranks = np.zeros((len(y_test), len(labels)))
		self.p_values = np.zeros((len(y_test), len(labels)))
		self.mondrian = mondrian

	def fit(self, ncf_measure):
		if self.mondrian:
			for l in self.labels:
				ncm_cal = ncf_measure(self.y_cal_proba, self.y_cal)
				ncm_cal = ncm_cal[self.y_cal == l]
				self.alphas = ncm_cal
				array_of_trues = np.full(len(self.y_test), l)  # Trick: run ncf_measure over *all* the test examples
				ncm_test = ncf_measure(self.y_test_proba, array_of_trues)
				sorted_alpha_cal = np.sort(ncm_cal)
				ranks = len(ncm_cal) - np.searchsorted(sorted_alpha_cal, ncm_test) + 1
				self.ranks[:, int(l)] = ranks
				p_value = ranks / (len(ncm_cal) + 1)
				self.p_values[:, int(l)] = p_value
		else:
			ncm_cal = ncf_measure(self.y_cal_proba, self.y_cal)
			self.alphas = ncm_cal
			for l in self.labels:
				array_of_trues = np.full(len(self.y_test), l)
				ncm_test = ncf_measure(self.y_test_proba, array_of_trues)
				sorted_alpha_cal = np.sort(ncm_cal)
				ranks = len(ncm_cal) - np.searchsorted(sorted_alpha_cal, ncm_test) + 1
				self.ranks[:, int(l)] = ranks
				p_value = ranks / (len(ncm_cal) + 1)
				self.p_values[:, int(l)] = p_value


	def point_prediction_performance(self, digits=2):
		y_hat_test = np.argmax(self.p_values, axis=1)
		print(classification_report(self.y_test, y_hat_test, digits=digits))

	def s_criterion(self):
		return np.average(self.p_values)

	def average_false_pvalue(self):
		y_test_false = np.logical_not(self.y_test).astype(int)
		false_p_values = np.take_along_axis(self.p_values, y_test_false[:, None], axis=1)
		return np.average(false_p_values)

	def top_high_confidence(self, n=10):
		second_largest = np.sort(self.p_values, axis=1)[:, -2]
		return np.argsort(second_largest)[:n]  # it's in increasing order

	def top_low_confidence(self, n=10):
		second_largest = np.sort(self.p_values, axis=1)[:, -2]
		return np.argsort(second_largest)[-n:]  # it's in increasing order

	def top_high_credibility(self, n=10):
		return np.argsort(np.max(self.p_values, axis=1))[-n:]

	def top_low_credibility(self, n=10):
		return np.argsort(np.max(self.p_values, axis=1))[:n]

	def plot_validity(self, labelwise=False):  # TODO: granularity;
		significance = []
		error_rates = []
		pd_labels = []

		if labelwise:
			labels = self.labels
		else:
			labels = ['all']
		
		for label in labels:	
			for i in range(0, 101, 2):
				eps = i/100
				significance.append(eps)
				if labelwise:
					regions = self.p_values[self.y_test==label, :] > eps
					true_label_not_present = np.invert(regions[np.arange(len(regions)), self.y_test[self.y_test==label]]) 
					error_rate = np.sum(true_label_not_present) / len(self.y_test[self.y_test==label])
				else:
					regions = self.p_values > eps
					true_label_not_present = np.invert(regions[np.arange(len(regions)), self.y_test]) 
					error_rate = np.sum(true_label_not_present) / len(self.y_test)
				error_rates.append(error_rate)
				pd_labels.append(str(label))

		df = pd.DataFrame({'significance':significance, 'error rate':error_rates, 'label':pd_labels})

		fig = px.scatter(df, x='significance', y='error rate', color='label', width=500, height=400,  # 700, 500
			template='none',
			color_discrete_sequence=px.colors.qualitative.T10
		)
		fig.add_trace(
			go.Scatter(x=significance,
				y=significance,
				opacity=0.2,
				line=dict(
					color='gray',
					dash='dash'
					),
				showlegend=False
				)
		)
		fig.update_layout(
			legend=dict(
			    yanchor="top",
			    y=0.94,
			    xanchor="left",
			    x=0.1),
			# font_family='Officina Sans ITC Book'
		)
		fig.update_xaxes(nticks=10)

		# fig = sns.scatterplot(data=df, x='significance', y='error rate', hue='label')

		return fig




	def old_plot_validity(self):
		error_rates = []
		significance = []
		sizes = []
		empties = []
		for i in range(0, 101, 2):
			eps = i/100
			significance.append(eps)
			regions = self.p_values > eps
			no_true_included = np.invert(np.take_along_axis(regions, self.y_test[:, None], axis=1))
			error_rate = np.sum(no_true_included) / len(self.y_test)
			error_rates.append(error_rate)
			region_size = np.sum(regions, axis=1)
			sizes.append(np.sum(region_size) / len(self.y_test))
			empties.append(np.sum(region_size == 0) / len(self.y_test))

		fig, axs = plt.subplots(3, sharex=True, figsize=(5,10))

		axs[0].plot(significance, significance, color='lightgray', linestyle='dashed')
		axs[0].plot(significance, error_rates)
		# ax[0].xlabel('Confidence')
		# ax[0].ylabel('Error rate')
		axs[0].legend(['Theoretical', 'Actual error rate'])
		# ax[0].rcParams.update({'font.size': 12})

		axs[1].plot(significance, sizes)
		axs[1].legend(['Average prediction size'])
		axs[2].plot(significance, empties)
		axs[2].legend(['Empty predictions'])

		for ax in axs.flat:
			ax.grid(b=True, color='#999999', linestyle='-', alpha=0.2)

		axs[2].set(xlabel='significance level')


