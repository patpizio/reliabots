import unittest

import sys
sys.path.insert(0, '../src/')

from conformal_predictors.icp import ConformalPredictor
from conformal_predictors.nc_measures import *
import conformal_predictors.calibrutils as cu

from sklearn.datasets import *
import numpy as np
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.base import clone
from sklearn.metrics import classification_report
from nonconformist.cp import IcpClassifier
from nonconformist.nc import NcFactory, InverseProbabilityErrFunc, MarginErrFunc



class TestPValues(unittest.TestCase):

    def test_iris(self):
        is_smoothed = False

        # iris = load_iris()
        iris = load_breast_cancer()

        model = KNeighborsClassifier(n_neighbors=11)
        test_model = clone(model)

        idx = np.random.permutation(iris.target.size)
        idx_train, idx_cal, idx_test = idx[:50], idx[50:100], idx[100:]

        ## Nonconformist
        nc = NcFactory.create_nc(
            model,
            InverseProbabilityErrFunc()
            # MarginErrFunc()
            )
        icp = IcpClassifier(nc, smoothing=is_smoothed)  # Create an inductive conformal classifier

        # Fit the ICP using the proper training set
        icp.fit(iris.data[idx_train, :], iris.target[idx_train])

        # Calibrate the ICP using the calibration set
        icp.calibrate(iris.data[idx_cal, :], iris.target[idx_cal])

        nonconformist_p_values = icp.predict(iris.data[idx_test, :])


        ## Test model
        y_cal = iris.target[idx_cal]
        y_test = iris.target[idx_test]

        test_model.fit(iris.data[idx_train, :], iris.target[idx_train])

        y_cal_proba = test_model.predict_proba(iris.data[idx_cal, :])
        y_test_proba = test_model.predict_proba(iris.data[idx_test, :])

        icp = ConformalPredictor(y_cal_proba, y_cal, y_test_proba, y_test, smoothed=is_smoothed, mondrian=False)

        icp.fit(negative_logit)
        # icp.fit(margin_error_func)


        self.assertEqual(np.round(np.sum(nonconformist_p_values - icp.p_values), 12), 0)



    def test_breast_cancer(self):
        pass

if __name__ == '__main__':
    unittest.main()