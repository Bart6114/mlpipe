import os

import pytest

from mlpipe import Pipe, Segment

from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


X, y = samples_generator.make_classification(
    n_informative=5, n_redundant=0, random_state=42)

anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')



@pytest.fixture(scope='module')
def resource_tmp_file(request):
    tmp_filename = 'test_sklearn.pkl'
    def resource_tmp_file_teardown():
        try:
            os.remove(tmp_filename)
        except:
            pass
    request.addfinalizer(resource_tmp_file_teardown)
    return tmp_filename


def test_save_fitted_load_predict(resource_tmp_file):
    p = Pipe() +\
        Segment(anova_filter, 'anova') +\
        Segment(clf, 'svc')

    p.fit(X,y)
    p._save(resource_tmp_file)
    p = Pipe._load(resource_tmp_file)
    p.predict(X)

