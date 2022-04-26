import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import log_loss, brier_score_loss


def reliabubble_plot(scores, y_test, num_bins=10, title=None):
	# Create 10 bins (n. 1 to n. 10) such that 0.0 and 1.0 values are included
	# Bin n. 0 would get values < 0.0 (it shouldn't happen).
	bins = np.linspace(0, 1 + 1e-8, num=num_bins + 1)  
	whichbin = np.digitize(scores, bins)  # [bin of example forall examples]
	counts = np.bincount(whichbin, minlength=len(bins))
	proportions = counts / len(y_test)

	true_per_bin = np.bincount(whichbin, weights=y_test, minlength=len(bins))

	accuracy = np.zeros(len(bins))  # Initialise to use the "where" condition in np.divide()
	accuracy = np.divide(true_per_bin, counts, where=counts!=0)

	score_sum_per_bin = np.bincount(whichbin, weights=scores, minlength=len(bins))

	confidence = np.zeros(len(bins))
	confidence = np.divide(score_sum_per_bin, counts, where=counts!=0)

    # 2 equally large bubbles on 2 different plots may reflect different numbers
    # this is due to plotly using relative frequencies to determine the bubble size (understandably)
    # how can we link two graphs' relative  frequencies so that it's easy to compare them? facets?
	fig = px.scatter(x=confidence, y=accuracy, size=counts, width=700, height=500,
		labels={
			'x':'probability',
			'y':'accuracy'
		},
		title = title)
	fig.add_trace(
		go.Scatter(x=bins,
			y=bins,
			opacity=0.2,
			line=dict(
				color='gray',
				dash='dash'
				),
			showlegend=False
			)
		)
	return fig


def calibration_errors(scores, y_test, num_bins=10, include_log_loss=True):
	errors = {}
	bins = np.linspace(0, 1 + 1e-8, num=num_bins + 1)
	whichbin = np.digitize(scores, bins)  # [bin of example forall examples]
	counts = np.bincount(whichbin, minlength=len(bins))
	assert counts[0] == 0, 'There seems to be a score < 0.0 somewhere.'
	proportions = counts / len(y_test)
	nonzero = counts != 0

	true_per_bin = np.bincount(whichbin, weights=y_test, minlength=len(bins))
	accuracy = np.divide(true_per_bin[nonzero], counts[nonzero])

	score_sum_per_bin = np.bincount(whichbin, weights=scores, minlength=len(bins))
	confidence = np.divide(score_sum_per_bin[nonzero], counts[nonzero])  # mean confidence per bin

	ece = np.sum(np.multiply(proportions[nonzero], np.abs(confidence - accuracy)))
	mce = np.max(np.abs(confidence - accuracy))

	errors['ECE'] = ece
	errors['MCE'] = mce
	if include_log_loss == True:
		errors['log loss'] = log_loss(y_test, scores)
	errors['Brier loss'] = brier_score_loss(y_test, scores)

	return errors

