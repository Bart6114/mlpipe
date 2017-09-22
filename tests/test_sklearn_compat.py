from mlpipe import Pipe, Segment

from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


X, y = samples_generator.make_classification(
    n_informative=5, n_redundant=0, random_state=42)

anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')


def test_sklearn_pipe_fit():
    p = Pipe() +\
        Segment(anova_filter, 'anova') +\
        Segment(clf, 'svc')

    p.fit(X,y)


def test_sklearn_pipe_fit_predict1():
    p = Pipe() +\
        Segment(anova_filter, 'anova') +\
        Segment(clf, 'svc')

    p.fit(X,y)
    p.predict(X)


def test_sklearn_pipe_fit_predict1_dummy():

    def dummy(*args):
        return args

    p = Pipe() +\
        Segment(anova_filter, 'anova') +\
        Segment(dummy) +\
        Segment(clf, 'svc')

    p.fit(X,y)
    p.predict(X, y=None)
