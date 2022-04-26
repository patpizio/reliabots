import numpy as np
from icp import ConformalPredictor
from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score, precision_recall_fscore_support, f1_score
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ideally:
class CrossConformalPredictor():
	def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, train_func, pred_func, n_folds, mondrian=False):
		pass

class CrossConformalPredictor():

	def __init__(self, X_train, y_train, X_test, y_test, validation_set, model_name, tokenizer, labels, n_folds=3, mondrian=True):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.validation_set = validation_set
		self.model_name = model_name
		self.tokenizer = tokenizer
		self.labels = labels
		self.num_labels = len(self.labels)
		self.p_values = np.zeros((len(y_test), len(labels)))
		self.n_folds = n_folds
		self.mondrian = mondrian
		self.proper_trains = DatasetDict()
		self.calis = DatasetDict()
		self.preds = {}
		kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=1986)
		for f, (pt, cal) in enumerate(kf.split(data['train'])):
			    self.proper_trains[f] = X_train.select(pt)
			    self.calis[f] = X_train.select(cal)



	def train(self, train_args):
		for i in range(self.n_folds):
		    preds[i] = {}

		    model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
		    model.to(device)

		    trainer = Trainer(
		        model,
		        args,
		        train_dataset=proper_trains[i],
		        eval_dataset=self.validation_set,
		        tokenizer=self.tokenizer,
		        compute_metrics=compute_metrics
		    )
		    trainer.train()
		    
		    # conformal prediction
		    y_cal = calis[i]['label'].numpy()
		    y_test = data['test']['label'].numpy()
		    cal_predictions = trainer.predict(calis[i])
		    y_cal_proba = cal_predictions.predictions
		    y_test_proba = trainer.predict(data['test']).predictions
		    labels = [i for i in range(self.num_labels)]
		    
		    preds[i]['y_cal'] = y_cal
		    preds[i]['y_cal_proba'] = y_cal_proba
		    preds[i]['y_test'] = y_test
		    preds[i]['y_test_proba'] = y_test_proba


	def compute_metrics(eval_pred):
	    predictions, labels = eval_pred
	    preds = np.argmax(predictions, axis=1)
	    precision, recall, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
	    acc = accuracy_score(labels, preds)
	    f1_weighted = f1_score(labels, preds, average='weighted')
	    return {
	        'accuracy': acc,
	        'f1_macro': f1_macro,
	        'f1_weighted': f1_weighted,
	        'precision': precision,
	        'recall': recall
	    }



	def fit(self, ncf_measure):
		for i in range(n_folds):
		    preds[i]['icp'] = ConformalPredictor(preds[i]['y_cal_proba'], preds[i]['y_cal'], 
		                                      preds[i]['y_test_proba'], preds[i]['y_test'], 
		                                      self.labels, mondrian=self.mondrian)
		    preds[i]['icp'].fit(ncf_measure)
		    preds[i]['ranks'] = preds[i]['icp'].ranks - 1  # necessary for cross-conformal

		    rz = np.sum(np.array([preds[i]['ranks'] for i in range(self.n_folds)]), axis=0)  # sum ranks over all folds

		    if is_mondrian:
			    for label in labels:
			        self.p_values[:, label] = (rz[:, label] + 1) / (int(sum(X_train['label'] == label)) + 1)
			else:
			    self.p_values = (rz + 1) / (len(self.X_train) + 1)


	def point_prediction_performance(self, digits=2):
		y_hat_test = np.argmax(self.p_values, axis=1)
		print(classification_report(self.y_test, y_hat_test, digits=digits))

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
